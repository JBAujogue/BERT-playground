import re
from itertools import chain
from typing import Any, Iterable

from unidecode import unidecode

from bertools.tasks.ner.typing import Record

# build a table mapping all non-printable ascii characters to None
# using 128 for ascii-only chars, for all unicode chars use "sys.maxunicode + 1" instead
NOPRINT_TRANS_TABLE = {i: None for i in range(128) if not chr(i).isprintable()}


def preprocess_records(records: Iterable[Record]) -> list[Record]:
    """
    Apply preprocessing steps on each line:
    - split content into printable ascii words
    - sort by decreasing number of words (context included).
    """
    records = split_into_printable_ascii_words(records)
    records = sort_by_length(records)
    return records


def split_into_printable_ascii_words(records: Iterable[Record]) -> list[Record]:
    """
    Split a string into a list of words.
    Each word is converted to printable ascii, and is dropped if conversion results in a
    blank string.
    """
    pattern = re.compile(r"\S+")
    return [split_content(r, pattern) for r in records]


def split_content(record: Record, pattern: re.Pattern[str]) -> Record:
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


def sort_by_length(records: list[Record]) -> list[Record]:
    """
    Sort lines by decreasing number of words.
    """
    return sorted(records, key=lambda r: len(r["words"]), reverse=True)


def concat_lists(ls: list[list[Any]]) -> list[Any]:
    """
    Flatten a list of list into a flat list.
    """
    return list(chain.from_iterable(ls))
