from dataclasses import dataclass, field
from typing import List, Tuple, Literal


@dataclass(kw_only=True)
class Shape:
    type: str  # "star", "circle", "square", "triangle"
    color: str
    size_category: Literal["small", "large"]
    x: float
    y: float
    pixel_size: float


@dataclass(kw_only=True)
class Star(Shape):
    type: str = "star"


@dataclass(kw_only=True)
class Distractor(Shape):
    pass


@dataclass
class QAPair:
    question: str
    answer: str


@dataclass
class SceneMetadata:
    width: int
    height: int
    difficulty: str
    stars: List[Star]
    distractors: List[Distractor]
    qa_pairs: List[QAPair] = field(default_factory=list)

