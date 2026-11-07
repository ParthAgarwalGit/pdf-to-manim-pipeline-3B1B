"""Professor directives: a free-form brief compiled into binding instructions.

Produced by ``ProfessorInputInterpreter`` from the brief given with the PDF
(Part F). Saved as ``brief_interpretation.json`` and shown back to the
professor before anything is spent. After scripting, a deterministic
compliance pass checks each directive; unmet ones are reported in the
delivery report. Nothing in the brief is dropped: whatever maps to no
structured field is kept verbatim in ``notes``.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ConceptDirective(BaseModel):
    concept: str
    depth: Optional[Literal["skip", "brief", "standard", "deep"]] = None
    analogy_brief: str = Field(
        default="", description="The professor's own analogy, kept verbatim"
    )
    avoid_analogies: List[str] = Field(
        default_factory=list, description="Analogies the professor rejected, e.g. 'clock'"
    )
    notes: str = ""


class NotationDirective(BaseModel):
    use: str
    instead_of: Optional[str] = None
    note: str = ""


class ScopeDirective(BaseModel):
    skip_sections: List[str] = Field(
        default_factory=list, description="Section labels/headings to leave out, e.g. '4.3'"
    )
    skip_pages: List[int] = Field(default_factory=list)
    only_pages: List[int] = Field(
        default_factory=list, description="When set, keep only these pages"
    )


class GlobalDirective(BaseModel):
    target_minutes: Optional[float] = None
    audience: Optional[Literal["high-school", "undergraduate", "graduate"]] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None
    tone: str = ""
    terminology: Dict[str, str] = Field(default_factory=dict)


class PresenterDirective(BaseModel):
    bridges: Optional[Literal["on", "off"]] = None
    layout: Optional[Literal["full", "pip"]] = None
    notes: str = ""


class ProfessorDirectives(BaseModel):
    concepts: List[ConceptDirective] = Field(default_factory=list)
    scope: ScopeDirective = Field(default_factory=ScopeDirective)
    notation: List[NotationDirective] = Field(default_factory=list)
    global_: GlobalDirective = Field(default_factory=GlobalDirective, alias="global")
    presenter: PresenterDirective = Field(default_factory=PresenterDirective)
    notes: List[str] = Field(
        default_factory=list,
        description="Brief content that maps to no structured field, verbatim",
    )
    source_brief: str = ""

    model_config = {"populate_by_name": True}

    def avoided_analogies(self) -> List[str]:
        return [a for c in self.concepts for a in c.avoid_analogies]

    def script_instructions(self) -> List[str]:
        """Binding instructions for the script prompt, in plain language."""
        lines: List[str] = []
        for c in self.concepts:
            if c.depth == "skip":
                lines.append(f"Do not cover {c.concept}.")
            elif c.depth in ("brief", "deep"):
                lines.append(
                    f"Cover {c.concept} "
                    + ("briefly." if c.depth == "brief" else "in depth, with an extra worked step or example.")
                )
            if c.analogy_brief:
                lines.append(f"For {c.concept}, use the professor's analogy: {c.analogy_brief}")
            for avoided in c.avoid_analogies:
                lines.append(f"Never use the {avoided} analogy" + (f" for {c.concept}." if c.concept else "."))
            if c.notes:
                lines.append(f"About {c.concept}: {c.notes}")
        for n in self.notation:
            lines.append(
                f"Notation: write {n.use}" + (f", not {n.instead_of}" if n.instead_of else "")
                + (f" ({n.note})" if n.note else "") + "."
            )
        g = self.global_
        if g.tone:
            lines.append(f"Tone: {g.tone}.")
        for term, preferred in g.terminology.items():
            lines.append(f"Say '{preferred}' rather than '{term}'.")
        lines.extend(self.notes)
        return lines

    def readback(self) -> List[str]:
        """'Here's how I understood your brief', one line per directive."""
        lines: List[str] = []
        s, g = self.scope, self.global_
        if s.skip_sections:
            lines.append(f"Leave out section(s): {', '.join(s.skip_sections)}")
        if s.skip_pages:
            lines.append(f"Leave out page(s): {', '.join(map(str, s.skip_pages))}")
        if s.only_pages:
            lines.append(f"Only use pages {s.only_pages[0]}-{s.only_pages[-1]}")
        if g.target_minutes:
            lines.append(f"Keep it under {g.target_minutes:g} minutes")
        if g.audience:
            lines.append(f"Audience: {g.audience}")
        if g.verbosity:
            lines.append(f"Verbosity: {g.verbosity}")
        if self.presenter.bridges:
            lines.append(f"Presenter bridges: {self.presenter.bridges}")
        lines.extend(self.script_instructions())
        return lines

    def is_empty(self) -> bool:
        return not (
            self.concepts or self.notation or self.notes
            or self.scope.skip_sections or self.scope.skip_pages or self.scope.only_pages
            or self.global_.target_minutes or self.global_.audience or self.global_.verbosity
            or self.global_.tone or self.global_.terminology
            or self.presenter.bridges or self.presenter.layout
        )
