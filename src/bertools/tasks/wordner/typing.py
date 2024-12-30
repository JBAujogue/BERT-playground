from typing import TypedDict, get_type_hints


class Span(TypedDict):
    """
    Base class for a span, eg an occurence of a category.
    """
    start: int
    end: int
    label: str
    text: str
    confidence: float


class Input(TypedDict):
    """
    Base class for an input of word-level NER model.
    """
    id: str
    content: str


class Output(TypedDict):
    """
    Base class for an output of word-level NER model.
    """
    id: str
    spans: list[Span]


class Record(TypedDict, total = False):
    """
    Base class for an output of word-level NER model.
    """
    id: str
    text_id: str
    line_id: str
    content: str
    words: list[str]
    offsets: list[tuple[int, int]]
    context: list[str]
    indices: list[int]
    confidences: list[float]
    spans: list[Span]


LINE_FIELDS = ['text_id', 'line_id', 'id', 'content']
SPAN_FIELDS = list(get_type_hints(Span).keys())
SPAN_GATHERING_FIELDS = ['spans']
