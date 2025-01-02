import re
from itertools import chain
from typing import Any, Iterable

from loguru import logger
from unidecode import unidecode

from bertools.tasks.wordner.utils.typing import Input, Record

# build a table mapping all non-printable ascii characters to None
# using 128 for ascii-only chars, for all unicode chars use "sys.maxunicode + 1" instead
NOPRINT_TRANS_TABLE = {i: None for i in range(128) if not chr(i).isprintable()}


def preprocess_records(
    inputs: Iterable[Input | Record],
    max_context_lines: int,
    prefix: str = "",
    suffix: str = "",
) -> list[Record]:
    """
    Apply preprocessing steps on each line:
    - split content into printable ascii words
    - append prefix / suffix str supplied by 'prefix' and 'suffix' params.
      Remark: these words are recognizable as those where the start offset equals
      the end offset.
    - build context of past lines specified by the 'max_context_lines' param.
    - sort by decreasing number of words (context included).
    """
    records = split_into_printable_ascii_words(inputs)
    records = append_boundary_words(records, prefix, suffix)
    records = build_context(records, max_context_lines)
    records = sort_by_length(records)
    return records


def split_into_printable_ascii_words(records: Iterable[Input | Record]) -> list[Record]:
    """
    Split a string into a list of words.
    Each word is converted to printable ascii, and is dropped if conversion results in a
    blank string.
    """
    pattern = re.compile(r"\S+")
    return [split_content(r, pattern) for r in records]


def split_content(record: Input | Record, pattern: re.Pattern[str]) -> Record:
    """
    Split string according to the supplied regex pattern, convert each split into
    printable ascii or remove it when conversion returns an empty string.
    """
    results = ((to_printable_ascii(m.group()), (m.start(), m.end())) for m in pattern.finditer(record["content"]))
    filtered = [m for m in results if len(m[0]) > 0]
    return Record(**record, words=[m[0] for m in filtered], offsets=[m[1] for m in filtered])


def to_printable_ascii(s: str) -> str:
    """
    Strip and filter out non-printable, non-ascii characters from input string.
    """
    return unidecode(s).strip().translate(NOPRINT_TRANS_TABLE)


def append_boundary_words(records: list[Record], prefix: str, suffix: str) -> list[Record]:
    """
    Append a prefix and a suffix str to each line in a list.
    """
    pref = to_printable_ascii(prefix)
    suff = to_printable_ascii(suffix)
    if prefix and not pref:
        logger.warning(f"Prefix {prefix} is discarded, as it contains no printable ascii character.")
    if suffix and not suff:
        logger.warning(f"Suffix {suffix} is discarded, as it contains no printable ascii character.")
    return [append_boundary_words_to_record(r, pref, suff) for r in records]


def append_boundary_words_to_record(record: Record, prefix: str, suffix: str) -> Record:
    """
    Append a prefix and a suffix str to a single line.
    Prefix have offset (0, 0).
    Suffix have offset (l, l) where l is the end char of the last word of the libe.
    """
    if prefix:
        record["words"] = [prefix] + record["words"]
        record["offsets"] = [(0, 0)] + record["offsets"]
    if suffix:
        length = max([0] + [o[1] for o in record["offsets"]])
        record["words"] += [suffix]
        record["offsets"] += [(length, length)]
    return record


def build_context(records: list[Record], max_context_lines: int) -> list[Record]:
    """
    Build left context by concatenating past lines from the text.
    """
    past = [r["words"] for r in records]
    return [r | {"context": concat_lists(past[max(0, i - max_context_lines) : i])} for i, r in enumerate(records)]


def sort_by_length(records: list[Record]) -> list[Record]:
    """
    Sort lines by decreasing number of words in context and content.
    """
    return sorted(records, key=lambda r: len(r["context"] + r["words"]), reverse=True)


def concat_lists(ls: list[list[Any]]) -> list[Any]:
    """
    Flatten a list of list into a flat list.
    """
    return list(chain.from_iterable(ls))
