from typing import TypedDict, get_type_hints


class Category(TypedDict):
    """
    Base class for a category from the taxonomy.
    """
    label_id: str
    label: str
    priority: int


class Span(Category):
    """
    Base class for a span, eg an occurence of a category.
    """
    start: int
    end: int
    text: str
    confidence: float


class SpanGathering(TypedDict):
    """
    Base class for a complete annotation of the toxicity over a message.
    """
    toxic: bool
    priority: int
    spans: list[Span]


class OptionalSpanGathering(TypedDict, total=False):
    """
    Base class for a complete annotation of the toxicity over a message.
    """
    toxic: bool
    priority: int
    spans: list[Span]


class Input(TypedDict):
    """
    Base class for an input of Toxbuster.
    """
    id: str
    content: str


class Output(SpanGathering):
    """
    Base class for an output of Toxbuster.
    """
    id: str
    version: str


SPAN_FIELDS = list(get_type_hints(Span).keys())
SPAN_GATHERING_FIELDS = list(get_type_hints(SpanGathering).keys())
