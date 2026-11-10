"""Structured edit operations on a lesson script (Part C of the design).

Professors edit meaning, not media: every change is one of these typed ops,
applied by ``ScriptEditorAgent`` to the lesson script. Natural-language
feedback is compiled into the same ops by ``ProfessorInputInterpreter``, so
the CLI, the UI and the interpreter share one deterministic apply path.
"""

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class RewriteBeat(BaseModel):
    """Exact professor text: the beat becomes ``author=professor, locked``."""
    op: Literal["rewrite_beat"] = "rewrite_beat"
    beat_id: str
    text: str = Field(..., min_length=1)


class ReviseBeat(BaseModel):
    """LLM rewrite of one beat following an instruction, neighbours as context."""
    op: Literal["revise_beat"] = "revise_beat"
    beat_id: str
    instruction: str = Field(..., min_length=1)


class ReplaceAnalogy(BaseModel):
    """Rewrite an analogy's introduction and every callback from a brief.

    Phrases the professor puts in double quotes are kept verbatim.
    """
    op: Literal["replace_analogy"] = "replace_analogy"
    analogy_id: str
    brief: str = Field(..., min_length=1)
    mapping: Dict[str, str] = Field(default_factory=dict)


class AddAnalogy(BaseModel):
    op: Literal["add_analogy"] = "add_analogy"
    concept: str
    brief: str = Field(..., min_length=1)
    after_beat: Optional[str] = None


class Remove(BaseModel):
    """Delete a beat, or an analogy with all of its callbacks."""
    op: Literal["remove"] = "remove"
    beat_id: Optional[str] = None
    analogy_id: Optional[str] = None

    @model_validator(mode="after")
    def _one_target(self) -> "Remove":
        if bool(self.beat_id) == bool(self.analogy_id):
            raise ValueError("remove needs exactly one of beat_id or analogy_id")
        return self


class Expand(BaseModel):
    """Insert deeper beats after ``target`` (a beat id or a concept label)."""
    op: Literal["expand"] = "expand"
    target: str
    levels: int = Field(default=1, ge=1, le=2)
    focus: str = "intuition"  # derivation | example | intuition | free text


class Condense(BaseModel):
    op: Literal["condense"] = "condense"
    beat_ids: List[str] = Field(..., min_length=2)


class Pace(BaseModel):
    """Pacing feedback: a pause after a beat. Text is unchanged."""
    op: Literal["pace"] = "pace"
    beat_id: str
    pause_seconds: float = Field(default=1.5, ge=0.0, le=10.0)


class Pronounce(BaseModel):
    """Audio feedback: how TTS should say a word (pronunciation lexicon)."""
    op: Literal["pronounce"] = "pronounce"
    word: str = Field(..., min_length=1)
    say_as: str = Field(..., min_length=1)


class FixVisual(BaseModel):
    """Visual feedback: regenerate the scene's code with this instruction.

    The narration (and its audio) is unchanged.
    """
    op: Literal["fix_visual"] = "fix_visual"
    beat_id: str
    instruction: str = Field(..., min_length=1)


class SetDensity(BaseModel):
    """Raise or lower concept density (Part D)."""
    op: Literal["set_density"] = "set_density"
    scope: Literal["global", "segment", "concept"] = "global"
    target: Optional[str] = None  # segment id or concept label for narrower scopes
    level: Literal["essential", "standard", "rich"] = "standard"
    analogies: Literal["on", "off"] = "on"


class PresenterOp(BaseModel):
    """Add, remove, edit or approve a presenter bridge (Part E)."""
    op: Literal["presenter"] = "presenter"
    action: Literal["insert", "remove", "edit", "approve"]
    beat_id: Optional[str] = None       # the bridge (remove/edit/approve) or anchor (insert)
    text: Optional[str] = None          # edit/insert text; insert/edit locks the bridge
    position: Literal["before", "after"] = "after"


EditOp = Annotated[
    Union[
        RewriteBeat, ReviseBeat, ReplaceAnalogy, AddAnalogy, Remove, Expand,
        Condense, Pace, Pronounce, FixVisual, SetDensity, PresenterOp,
    ],
    Field(discriminator="op"),
]

EDIT_OPS_ADAPTER = TypeAdapter(List[EditOp])

# How each op type invalidates artifacts (Part G feedback classes).
OP_CLASSES: Dict[str, str] = {
    "rewrite_beat": "content",
    "revise_beat": "content",
    "replace_analogy": "content",
    "add_analogy": "content",
    "remove": "content",
    "expand": "content",
    "condense": "content",
    "set_density": "content",
    "pace": "pacing",
    "pronounce": "audio",
    "fix_visual": "visual",
    "presenter": "presenter",
}


def parse_ops(data) -> List[EditOp]:
    """Validate a list of op dicts (from YAML/JSON or an LLM)."""
    if isinstance(data, dict) and "ops" in data:
        data = data["ops"]
    return EDIT_OPS_ADAPTER.validate_python(data or [])
