import sys
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from typing import Iterable, Optional, TextIO

import click
from babel.core import Locale
from babel.messages.catalog import Message
from babel.messages.pofile import read_po, write_po
from more_itertools import chunked
from openai import OpenAI, Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from termcolor import colored
from tqdm import tqdm


@click.command()
@click.option(
    "--model",
    default="gpt-4o",
    help="The OpenAI model to use",
)
@click.option(
    "--batch-size",
    default=50,
    help="The number of messages to translate at once",
)
@click.option(
    "--max-retries",
    default=3,
    help="The maximum number of retries for a message",
)
@click.option(
    "--fuzzy-only",
    is_flag=True,
    default=False,
    help="Only translate fuzzy messages",
)
@click.option(
    "--ignore-existing",
    is_flag=True,
    default=False,
    help="Ignore existing translations in .po file",
)
@click.option(
    "--inplace",
    is_flag=True,
    default=False,
    help="Modify .po file in place",
)
@click.option(
    "--knowledge",
    type=click.Path(),
    required=False,
    help="Extra information to provide to the model",
)
@click.argument("source_locale")
@click.argument("target_locale")
@click.argument("po_file", type=click.Path())
@click.argument("po_outfile", type=click.Path(), required=False)
def main(
    model: str,
    batch_size: int,
    max_retries: int,
    fuzzy_only: bool,
    ignore_existing: bool,
    inplace: bool,
    knowledge: Optional[str],
    source_locale: str,
    target_locale: str,
    po_file: str,
    po_outfile: Optional[str],
):
    locales = (Locale.parse(source_locale), Locale.parse(target_locale))

    with open(po_file) as f:
        catalog = read_po(f)

    if knowledge:
        with open(knowledge) as f:
            knowledge = f.read()

    translatables = [
        Translatable(message, ignore_existing=ignore_existing) for message in catalog
    ]

    def condition(t: Translatable) -> bool:
        if fuzzy_only:
            return (
                t.message.fuzzy
                or not t.message.string
                or t.message.id == t.message.string
            )
        else:
            return True

    for batch in chunked(filter(condition, translatables), batch_size):
        messages: dict[str, tuple[str, str]] = {}
        sources: dict[str, list[Translatable]] = defaultdict(list)

        for trans in batch:
            for key, source, target in trans:
                messages[key] = (source, target)
                sources[key].append(trans)

        translator = Translator(
            messages,
            locales,
            knownledge=knowledge,
            model=model,
            max_retries=max_retries,
        )

        for key, value in translator.translate():
            for origin in sources.get(key, []):
                origin[key] = value

    for trans in translatables:
        trans.finalize()

    buf = BytesIO()
    write_po(buf, catalog, width=0)

    if inplace:
        with open(po_file, "wb") as f:
            f.write(buf.getvalue())
    elif po_outfile:
        with open(po_outfile, "wb") as f:
            f.write(buf.getvalue())
    else:
        sys.stdout.write(buf.getvalue().decode("utf-8"))


class Translator:
    def __init__(
        self,
        messages: dict[str, tuple[str, str]],
        locales: tuple[Locale, Locale],
        *,
        knownledge: Optional[str] = None,
        model: str,
        max_retries: int = 3,
        stderr: TextIO = sys.stderr,
    ):
        self.messages = messages
        self.retries = defaultdict[str, int](int)
        self.source_locale, self.target_locale = locales
        self.knowledge = knownledge
        self.model = model
        self.max_retries = max_retries
        self.stderr = stderr
        self.client = OpenAI()

    def translate(self) -> Iterable[tuple[str, str]]:
        self.retries.clear()

        stream = self.client.chat.completions.create(
            messages=self.create_prompt(),
            model=self.model,
            temperature=0.2,
            stream=True,
        )

        while self.messages:
            for msgid, msgstr in self.digest(stream):
                self.messages.pop(msgid, None)
                self.retries.pop(msgid, None)
                yield msgid, msgstr

            for msgid in self.messages:
                self.retries[msgid] += 1

            for msgid, attempt in [*self.retries.items()]:
                if attempt > self.max_retries:
                    self.messages.pop(msgid, None)
                    self.retries.pop(msgid, None)

    def digest(self, stream: Stream[ChatCompletionChunk]) -> Iterable[tuple[str, str]]:
        translator = self

        class ExpectId(BaseModel):
            def next(self, line: str):
                message = translator.messages.get(line)
                if message is None:
                    return (colored(line or repr(line), "red"), ExpectId())
                else:
                    return (message[0], ExpectMessage(id=line))

        class ExpectMessage(BaseModel):
            id: str

            def next(self, line: str):
                line = line.strip()
                return (
                    colored(line or repr(line), "green") + "\n",
                    Success(id=self.id, message=line),
                )

        class Success(BaseModel):
            id: str
            message: str

            def next(self):
                return ExpectId()

        state: ExpectId | ExpectMessage = ExpectId()

        def mutate(line: str):
            nonlocal state

            outcome = state.next(line)

            sys.stderr.write(outcome[0] + "\n")
            sys.stderr.flush()

            if isinstance(outcome[1], Success):
                yield outcome[1].id, outcome[1].message
                state = ExpectId()

            else:
                state = outcome[1]

        incomplete_line = ""

        with tqdm(bar_format="{desc}", file=self.stderr, leave=False) as pbar:
            for chunk in stream_content(stream):
                for idx, text in enumerate(chunk.split("\n")):
                    if idx == 0:
                        incomplete_line += text
                        pbar.set_description(colored(incomplete_line, "grey"))
                        continue

                    pbar.set_description(None, False)
                    pbar.clear()

                    if idx == 1:
                        line = incomplete_line + text
                        incomplete_line = ""
                    else:
                        line = text

                    if line.isspace() or not line:
                        continue

                    yield from mutate(line)

            pbar.set_description(None, False)
            pbar.clear()

            if incomplete_line:
                yield from mutate(incomplete_line)

    def create_prompt(self) -> Iterable[ChatCompletionMessageParam]:
        task: list[str] = []

        for key, (source, target) in self.messages.items():
            task.append(key)
            task.append(source)
            task.append(target)

        return [
            {
                "role": "system",
                "content": "You are an expert in technical writing."
                " You are assisting the user with localization of technical documentation.",
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "# Task",
                        "",
                        f"Translate documentation from {self.source_locale.get_display_name()}"
                        f" to {self.target_locale.get_display_name()}.",
                        "",
                        "## Input format",
                        "",
                        "Text to be translated will be provided to you in the `text/plain` format.",
                        "",
                        "The number of lines MUST be a multiple of 3. There MUST be at least 2 lines.",
                        "",
                        "Every 3 lines denote a 'message', where:",
                        "",
                        "Line 1 is an identifier for a message to translate.",
                        " The identifier MUST be a non-empty hexadecimal string.",
                        " The identifier MUST be unique within the input.",
                        "",
                        "Line 2 is the source text for you to translate.",
                        " The source text MUST NOT contain any newlines.",
                        " The source text MUST NOT be empty.",
                        "",
                        "Line 3 is the existing translation of the source text.",
                        " The translation MUST NOT contain any newlines.",
                        " The translation MAY be empty.",
                        "",
                        "If an existing translation exists, you MUST correct it, including content and formatting.",
                        " You MUST remove any extraneous information not present in the source text",
                        " from an existing translation.",
                        "",
                        "## Output format",
                        "",
                        "You MUST reply in the `text/plain` format.",
                        "",
                        "The number of lines in your reply MUST be a multiple of 2. There MUST be at least 2 lines.",
                        "",
                        "Every 2 lines denote a 'message', where:",
                        "",
                        "Line 1 is an identifier for a message that you translated.",
                        " The identifier MUST exist in the input.",
                        "",
                        "Line 2 is your translation of the corresponding source text.",
                        " Your translation MUST NOT contain any newlines.",
                        "",
                        "Your reply MUST NOT contain content that is neither an identifier nor a translation.",
                        "",
                        "## Localization",
                        "",
                        "Source text MAY include markups in Markdown or reStructuredText. If encountered,",
                        " you MUST preserve the markups in the translation,",
                        " you MUST NOT introduce any new markups,",
                        " additionally, you MUST NOT translate text within markups that denote 'literal' content,",
                        " such as `inline code` or ```code blocks```.",
                        "",
                        "You MAY assume that messages appear in the order that they will appear in the final document.",
                        " As such, you MAY use context from previous and/or next messages to inform your translations.",
                        "",
                        "Whenever appropriate, you MAY rearrange the order of sentences or phrases across multiple messages,",
                        " provided that the overall meaning is preserved.",
                        "",
                        "## Typesetting - CJKV",
                        "",
                        "You SHOULD insert whitespace between CJKV characters and Latin characters.",
                        "",
                        "You SHOULD ensure that punctuation marks are correctly placed.",
                        "",
                        *(
                            [
                                "## Contextual info",
                                "",
                                "User has provided the following helpful information.",
                                "",
                                self.knowledge,
                            ]
                            if self.knowledge
                            else []
                        ),
                        "## Example input (from English to 中文（简体）)",
                        "",
                        "a826c7e389ec9f379cafdc544d7e9a4395ff7bfb58917bbebee51b3d0b1c996a",
                        "This is a message",
                        "",
                        "## Example output (from English to 中文（简体）)",
                        "",
                        "a826c7e389ec9f379cafdc544d7e9a4395ff7bfb58917bbebee51b3d0b1c996a",
                        "这是一条消息",
                        "",
                        "---",
                        "",
                        'End of instructions. Reply "Understood." verbatim to start the task.',
                        f" You will be translating from {self.source_locale.get_display_name()}"
                        f" to {self.target_locale.get_display_name()}.",
                        " The input will follow immediately after your reply.",
                    ]
                ),
            },
            {
                "role": "assistant",
                "content": "Understood.",
            },
            {
                "role": "user",
                "content": "\n".join(task),
            },
        ]


class Translatable:
    def __init__(self, message: Message, *, ignore_existing=False):
        self.message = message

        self.chunks: list[tuple[str, str, str]] = []
        self.updated = False

        if not isinstance(message.id, str):
            raise ValueError(f"Unsuported message id: {message.id}")
        if not isinstance(message.string, (str, type(None))):
            raise ValueError(f"Unsuported message string: {message.string}")

        trans_lines = message.string.split("\n") if message.string else []

        for idx, source in enumerate(message.id.split("\n")):
            key = sha256(source.encode()).hexdigest()[:7]
            try:
                if ignore_existing:
                    existing = ""
                else:
                    existing = trans_lines[idx]
            except IndexError:
                existing = ""
            self.chunks.append((key, source, existing))

    def __iter__(self):
        for key, source, target in self.chunks:
            if source:
                yield key, source, target

    def __setitem__(self, key, value):
        for idx, (k, source, target) in enumerate(self.chunks):
            if k == key:
                self.updated = True
                self.chunks[idx] = (k, source, value)
                return
        raise KeyError(key)

    def finalize(self):
        if not self.updated:
            return
        result = "\n".join([v for _, _, v in self.chunks])
        self.message.string = result
        self.message.flags.discard("fuzzy")


def stream_content(stream: Stream[ChatCompletionChunk]) -> Iterable[str]:
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    main()
