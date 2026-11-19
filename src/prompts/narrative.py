"""Prompts that decide what the video says: the lesson plan, the script
written from it, and the presenter bridges.

Order of calls: ``lesson_plan_prompt`` (one call, whole source) ->
``script_from_plan_prompt`` (one call, whole script) -> optionally
``bridge_prompt`` (presenter lines between acts).
"""

import re
from typing import Any, Dict, Iterable, List, Optional

# Pace used to turn a word budget into minutes for the planner and writer.
# Matches ScriptArchitectAgent.SPOKEN_WORDS_PER_SECOND (measured Nov 2026).
SPOKEN_WORDS_PER_SECOND = 2.8

_AUDIENCE_LINES = {
    "high-school": "They are bright but new to this: everyday comparisons, no jargon left undefined.",
    "graduate": "They are comfortable with abstraction: be precise and reach the general statement.",
}
_DEFAULT_AUDIENCE_LINE = (
    "They know the prerequisites but have not seen this topic: build intuition first, "
    "then state the result precisely."
)

SPOKEN_STYLE_RULES = """SPOKEN NARRATION (it is read aloud; it must sound like a person talking, not an AI or a textbook):
- Talk the way a good lecturer talks to one student at the board: plain words, contractions
  (it's, we'll, that's), mostly short sentences with the occasional longer one.
- One idea per sentence. No semicolons, parentheses, em dashes, lists or headings.
- Each beat picks up exactly where the previous one stopped. Use a connective only when it
  carries meaning ("so", "but", "which means"); never restart with a summary.
- Vary how beats begin. Never start two beats in a row with the same word, and do not lean on
  "So", "Now", "That", "This" or "What matters is" as openers.
- Never use stock phrases: "let's dive in", "here's the key", "here's the twist", "here's the thing",
  "there's a catch", "it turns out", "it's important to note", "as we can see", "let's take a look",
  "crucial", "fascinating", "in essence", "delve", "in this video", "simply put", "basically",
  "essentially". No exclamation marks. At most one question per act.
- Say math the way you would out loud: "A equals U times D times V transpose", "sigma one",
  "the length of A v". Never put symbols such as ^T, _k, ||x|| or LaTeX in the narration.
- Mention the picture only when the visual hint shows it ("watch the circle stretch into an ellipse").
"""

TYPED_BEAT_PROMPT_RULES = """BEAT TAGS:
- "role": core | analogy | example | derivation | intuition | recap.
  core = the essential argument; analogy = introduces or calls back an analogy;
  example = a worked instance; derivation = step-by-step working;
  intuition = an informal "why"; recap = a short summary.
- "depth": 0 when the beat is essential to the argument; 1-3 for increasingly
  optional elaboration.
- "analogy_id": id of the analogy this beat introduces or calls back, else null.
  Declare every new analogy in "analogies" (analogy_id, label, target_concepts,
  mapping from analogy term to concept). Callbacks reuse the existing id.
  Return "analogies": [] when there is no new analogy."""


# ── shared rendering helpers ────────────────────────────────────────────


def minutes_for_words(words: int) -> float:
    return round(words / SPOKEN_WORDS_PER_SECOND / 60.0, 1)


# A beat averages about this many words: planned scripts measured 25 to 29 per
# beat (Nov 2026), so asking for words/26 beats lands a script on its budget.
WORDS_PER_BEAT = 26
# A planned script below this share of its word budget is too short for the
# requested video length and is sent back to the writer once.
SCRIPT_FLOOR_SHARE = 0.92


def beats_for_words(words: int) -> int:
    """Beats an act needs to reach its word count."""
    return max(1, round(words / WORDS_PER_BEAT))


def act_count_range(word_budget: int) -> tuple:
    """(min, max) acts a budget can carry: about 130 words per act, 3 to 8 acts."""
    most = max(3, min(8, round(word_budget / 130)))
    least = 2 if word_budget < 220 else 3
    return least, max(least, most)


def render_source(segments: Iterable[Any], max_chars: int = 120_000) -> str:
    """The source as ``[seg_id] pages a-b | heading`` blocks, cut at ``max_chars``."""
    blocks: List[str] = []
    used = 0
    for segment in segments:
        heading = (getattr(segment, "heading", "") or "").strip() or "(no heading)"
        pages = f"{getattr(segment, 'page_start', '?')}-{getattr(segment, 'page_end', '?')}"
        text = (getattr(segment, "text", "") or "").strip()
        formulas = [f for f in (getattr(segment, "math_latex", None) or []) if f]
        block = f"[{segment.segment_id}] pages {pages} | {heading}\n{text}"
        if formulas:
            block += "\nformulas: " + " ; ".join(formulas[:12])
        if used + len(block) > max_chars:
            room = max_chars - used
            if room > 400:
                blocks.append(block[:room] + "\n(segment cut here: source too long)")
            blocks.append("(remaining segments not shown: source too long)")
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _brief_block(professor_notes: Optional[List[str]]) -> str:
    notes = [n for n in (professor_notes or []) if n]
    if not notes:
        return ""
    return (
        "\nPROFESSOR'S BRIEF (binding). Where the brief and the document disagree about what to "
        "include or how deep to go, the brief wins. A request to skip something applies to the "
        "whole video. A request to explain, show or use something is met once, where it fits best.\n"
        + "\n".join(f"- {note}" for note in notes)
        + "\n"
    )


def _concept_plan_block(concept_plan_lines: Optional[List[str]]) -> str:
    lines = [line for line in (concept_plan_lines or []) if line]
    if not lines:
        return ""
    return "\nCONCEPT PLAN (binding, per concept):\n" + "\n".join(f"- {line}" for line in lines) + "\n"


def _feedback_block(review_feedback: Optional[List[str]], what: str) -> str:
    items = [item for item in (review_feedback or []) if item]
    if not items:
        return ""
    return (
        f"\nA REVIEWER REJECTED THE PREVIOUS {what}. Fix every point:\n"
        + "\n".join(f"- {item}" for item in items[:8])
        + "\n"
    )


def _audience(audience: str) -> str:
    return f"{audience}. {_AUDIENCE_LINES.get(audience, _DEFAULT_AUDIENCE_LINE)}"


# ── 1. the lesson plan ──────────────────────────────────────────────────


def lesson_plan_prompt(
    segments: Iterable[Any],
    audience: str,
    word_budget: int,
    professor_notes: Optional[List[str]] = None,
    concept_plan_lines: Optional[List[str]] = None,
    presenter_opens: bool = False,
    review_feedback: Optional[List[str]] = None,
    max_source_chars: int = 120_000,
    length_requested: bool = True,
) -> str:
    """One call over the whole source: choose the question, the example, the acts.

    ``presenter_opens``: a presenter speaks the hook on camera before the
    first animation (hybrid runs with proposed bridges).
    ``length_requested``: False when nobody asked for a length and
    ``word_budget`` is only a default; it is then a ceiling, not a target.
    """
    least, most = act_count_range(word_budget)
    if length_requested:
        length_line = (
            f"LENGTH: {word_budget} words of narration in total, about "
            f"{minutes_for_words(word_budget)} minutes spoken. This is a hard budget: a plan that "
            "needs more words is the wrong plan."
        )
        words_rule = f"The words of all acts add up to {word_budget}."
        words_check = f"Do the acts' words add up to {word_budget}?"
    else:
        length_line = (
            f"LENGTH: nobody asked for a particular length. Use at most {word_budget} words of "
            f"narration (about {minutes_for_words(word_budget)} minutes), and fewer whenever the "
            "argument is complete in fewer: a tight video is better than a long one."
        )
        words_rule = (
            f"The words of all acts add up to at most {word_budget}; give each act only what its "
            "one idea needs."
        )
        words_check = "Is every act as short as its one idea allows?"
    hook_words = max(25, min(60, round(word_budget * 0.12)))
    payoff_words = max(40, min(90, round(word_budget * 0.12)))
    min_act_words = 40 if word_budget >= 300 else 25
    if presenter_opens:
        hook_rule = (
            "A presenter speaks \"hook\" on camera before any animation, so it must work with no "
            "picture: first person, at most 35 words, ending on the central question without "
            "answering it. The presenter's hook is NOT an act and uses none of the word budget. "
            "Act 1 is the animated scene that follows the presenter: it opens directly on the "
            "running example and makes the question concrete. Give it a real viewer_question and "
            "key_idea like every other act, never a placeholder."
        )
    else:
        hook_rule = (
            "\"hook\" is the first thing said, over the opening picture: at most 40 words, no "
            "greeting, no topic announcement, ending on the central question without answering it."
        )
    hook_rule += (
        " Write the hook the way a person talks: no stock phrases (\"it turns out\", \"turns out\", "
        "\"in this video\", \"let's dive in\"), no exclamation marks."
    )
    return f"""You are the story editor for a short animated explainer, the kind made by the best visual math and science channels. A document is attached. Your job is not to summarize it. Your job is to decide the ONE question this video answers and the shortest visual path from "never seen this" to "I see why". A writer will script from your plan and animators will draw what you describe, so every field must be concrete enough to act on.

AUDIENCE: {_audience(audience)}
{length_line}
{_brief_block(professor_notes)}{_concept_plan_block(concept_plan_lines)}{_feedback_block(review_feedback, "PLAN'S SCRIPT")}
SOURCE DOCUMENT, in segments. This is the document's order, which is rarely the right order for a video:
<<<
{render_source(segments, max_source_chars)}
>>>

HOW TO PLAN

1. AUDIT. List the document's distinct ideas. For each: what kind of thing it is (definition, result, proof, method, example, application, aside) and which segments hold it. Do this before deciding anything else.

2. ONE CENTRAL QUESTION. Choose the question a curious viewer would actually ask, in their words, that the document's main result answers. It must be answerable with a picture. A topic is not a question.
   Weak: "What is X?", "An overview of X", "Understanding X".
   Strong: a question with tension in it. Something looks impossible, wasteful or surprising until the document's main idea explains it.
   Keep it to one sentence of at most 25 words that a newcomer could repeat.

3. RUNNING EXAMPLE. Pick one small concrete case with actual numbers that the whole video keeps returning to. It must be drawable on a flat screen: two dimensions, a handful of points, single-digit numbers. Every general statement will first be seen happening to this example. Give its actual values. Also list under "values" the symbols the video will show on screen for its main objects, written exactly as the brief (or, failing that, the source) writes them, for instance the letters of a factorization, so every scene labels them the same way.

4. VISUAL MOTIFS. Choose 2 to 4 recurring on-screen objects built from the running example. Give each a fixed color and a fixed meaning; the viewer learns the picture once and it never changes its look. Colors to choose from: BLUE_B, TEAL_C, GREEN_C, YELLOW_C, GOLD_C, RED_C, MAROON_C, PURPLE_C, WHITE, GRAY_B. Motifs are drawn with simple shapes: dots, arrows, lines, curves, grids, circles, rectangles and short labels.

5. ACTS, in viewing order. Between {least} and {most} acts, and never more than {most}: at this length each extra act takes words from the others until none of them has room to show anything. Merge ideas that share one picture into one act.
   - Act 1 is the hook (kind "hook"). It opens on the running example doing something that raises the central question: a puzzle, a surprising outcome, or a task that looks hard. No definitions, no history, no "in this video". {hook_rule}
   - Every later act answers exactly one question the previous act left the viewer with ("viewer_question") and adds exactly one new idea ("key_idea", one plain sentence).
   - Show before you name. An idea first appears as something visibly happening to the running example. Its name and its formula come afterwards, as a label for what was just seen.
   - Order by need, not by the document. Put each idea where the viewer first needs it. Cut definitions the argument never uses.
   - "visual" says what the viewer WATCHES CHANGE in this act, in terms of the motifs: what moves, stretches, rotates, splits, lines up or collapses. Never what text appears, and never a person talking: every act is animated.
   - The last act is the payoff (kind "payoff"): it returns to the hook's picture and answers the central question. Its key_idea is the takeaway. It is short, because it says the answer once and stops.
   - "words" is the act's share of the budget. The hook gets at most {hook_words} words and the payoff at most {payoff_words}; no act gets fewer than {min_act_words}. {words_rule}
   - Accuracy beats neatness. Every key_idea must be true as stated, for the running example and in general. When a tidy connection between two pictures holds only in a special case, plan the weaker statement that is always true.
   - "source_segments" lists the segments the act takes its facts from.

6. COVERAGE. Give EVERY segment exactly one treatment:
   - covered: an act explains it.
   - compressed: an act states its conclusion in a sentence or one picture, without the working.
   - deferred: named in passing as something beyond this video.
   - omitted: left out. Give the reason in one sentence.
   A short video cannot carry a whole chapter. Within this budget it is correct to omit side results and methods that do not serve the central question, and to say so here.
   Formal proofs: unless the brief asks for proofs, keep only the one idea that makes a result believable, shown as a picture inside the act that needs it, and mark the proof's own segments compressed or omitted. Never plan an act whose job is to walk through the steps of a proof.

CHECK BEFORE YOU ANSWER
- Could a viewer say what the central question is after the first twenty seconds?
- Does every act's visual use the running example or a motif?
- Read the key_ideas in order. Do they form one argument in which each step needs the one before it?
- Does the last act answer the central question?
- Does every segment id above appear once in "coverage"?
- {words_check}

Respond with one JSON object in exactly this shape (no other text):
{{
  "audit": [{{"idea": "...", "kind": "definition|result|proof|method|example|application|aside", "segments": ["seg_001"]}}],
  "title": "a short title, not a question",
  "central_question": "...",
  "hook": "...",
  "promise": "one sentence: what the viewer will be able to see or explain by the end",
  "takeaway": "the answer to the central question in one sentence",
  "running_example": {{"description": "...", "values": {{"name": "actual value"}}}},
  "motifs": [{{"motif_id": "snake_case_name", "what": "what is drawn", "color": "BLUE_B", "stands_for": "the idea it represents"}}],
  "acts": [{{"kind": "hook|build|example|payoff|application|recap", "viewer_question": "...", "key_idea": "...", "visual": "...", "source_segments": ["seg_001"], "words": 60}}],
  "coverage": [{{"segment_id": "seg_001", "treatment": "covered|compressed|deferred|omitted", "reason": "..."}}]
}}
"""


# ── 2. the script, written from the plan ────────────────────────────────


def render_plan(plan: Any) -> str:
    """The plan as the writer, bridge writer and grader read it."""
    lines = [
        f"Title: {plan.title}",
        f"Central question: {plan.central_question}",
        f"Hook: {plan.hook}",
        f"Promise: {plan.promise}",
        f"Takeaway (the last thing the viewer hears): {plan.takeaway}",
    ]
    example = plan.running_example
    if example.description or example.values:
        lines.append(f"Running example: {example.description}")
        for name, value in example.values.items():
            lines.append(f"  {name} = {value}")
    if plan.motifs:
        lines.append("Motifs (use these names in visual hints):")
        for motif in plan.motifs:
            lines.append(
                f"  {motif.motif_id} ({motif.color or 'no fixed color'}): {motif.what}"
                + (f" - stands for {motif.stands_for}" if motif.stands_for else "")
            )
    lines.append("Acts, in order:")
    for act in plan.acts:
        lines.append(
            f"  {act.act_id} [{act.kind}, about {act.words} words in about "
            f"{beats_for_words(act.words)} beats] "
            f"viewer question: {act.viewer_question or '(none)'} | key idea: {act.key_idea} | "
            f"visual: {act.visual or '(writer chooses)'} | facts from: "
            f"{', '.join(act.source_segments) or '(none)'}"
        )
    left_out = [c for c in plan.coverage if c.treatment in ("compressed", "deferred", "omitted")]
    if left_out:
        lines.append("Deliberately not explained in full (do not bring the working back):")
        for decision in left_out:
            lines.append(f"  {decision.segment_id}: {decision.treatment}. {decision.reason}")
    return "\n".join(lines)


def script_from_plan_prompt(
    plan: Any,
    segments: Iterable[Any],
    audience: str,
    word_budget: int,
    professor_notes: Optional[List[str]] = None,
    concept_plan_lines: Optional[List[str]] = None,
    presenter_opens: bool = False,
    review_feedback: Optional[List[str]] = None,
    max_source_chars: int = 120_000,
    previous_draft: str = "",
    length_requested: bool = True,
) -> str:
    """One call: the whole narration, beat by beat, following the plan's acts.

    ``previous_draft`` (``draft_listing`` of a rejected attempt) is shown to
    the writer on a retry so it extends that draft instead of starting over.
    ``length_requested``: False when nobody asked for a length; the word
    counts are then ceilings, and the writer is told not to fill them.
    """
    floor_words = round(word_budget * SCRIPT_FLOOR_SHARE)
    total_beats = sum(beats_for_words(act.words) for act in plan.acts)
    if length_requested:
        length_rules = (
            "- Write the acts in order and tag every beat with its \"act_id\". Every act in the plan "
            "gets beats. The plan says how many beats each act needs to reach its word count (about "
            f"{total_beats} beats in all); each act lands within 15 percent of its words.\n"
            f"- The whole script is between {floor_words} and {word_budget} words. The video has a "
            "set length: do not run over, and do not come in short. Count each act's words before "
            "you answer.\n"
            "- An act that comes out short is missing a step the viewer needs. Add that step: one "
            "more thing that happens to the running example, shown and said. Never pad a sentence "
            "to make length."
        )
    else:
        length_rules = (
            "- Write the acts in order and tag every beat with its \"act_id\". Every act in the plan "
            "gets beats. The plan's word counts are ceilings: an act may use fewer words, never "
            "more.\n"
            f"- The whole script is at most {word_budget} words. Nobody asked for a particular "
            "length, so that is a ceiling and not a target: once an act has shown and said its key "
            "idea, move on. Never add a sentence, an example or a restatement to make length."
        )
    draft_block = (
        "\nYOUR PREVIOUS DRAFT (rejected; see the reviewer's points above). Keep what works in "
        "it and rewrite the whole script:\n" + previous_draft.strip() + "\n"
        if previous_draft.strip() else ""
    )
    if presenter_opens:
        opening_rule = (
            f"A presenter has already said the hook on camera: \"{plan.hook}\" Do not repeat it. "
            "Your first beat starts on the picture and goes straight to work on the question."
        )
    else:
        opening_rule = (
            "The first beat is the hook, spoken over the opening picture. Use the plan's hook nearly "
            "as written. No greeting, no topic announcement, no definitions."
        )
    return f"""You are writing the narration for a short animated explainer. One person speaks it over animation. The plan below is settled: its question, its running example, its acts and their order are not yours to change. Your job is the spoken words and, for every beat, what the viewer watches while hearing them.

THE PLAN (binding)
{render_plan(plan)}

AUDIENCE: {_audience(audience)}
{_brief_block(professor_notes)}{_concept_plan_block(concept_plan_lines)}{_feedback_block(review_feedback, "SCRIPT")}{draft_block}
SOURCE (for facts, definitions and formulas only; ignore its order, and add nothing the plan left out):
<<<
{render_source(segments, max_source_chars)}
>>>

HOW TO WRITE IT

Beats
- A beat is one visual moment: one to three sentences, 20 to 45 words, never more than 55. When the picture has to change twice, that is two beats.
{length_rules}

Economy (this is what separates a script from a transcript of a lecture)
- Say each thing once. Every sentence either tells the viewer something new or tells them where to look.
- No warm-up and no announcements. Never say what you are about to explain, never recap what was just shown, never open a beat by restating the previous one.
- Let the picture carry the detail. Do not read out coordinates, matrix entries or every term of a formula that is on screen. Say what to notice about them.
- Motif names such as best_fit_line are labels for the animators and appear in "visual_hint" only. The narration never contains a name with an underscore: say what the thing is in plain words ("the gold line", "the cloud of points").
- Prefer the short concrete word: "stretch" rather than "apply a scaling transformation".

Shape
- Each act opens by making its viewer question felt, as a question, a snag or a "but", never as a heading. It ends once its key idea has been said plainly, one time.
- Show before naming. Describe what happens to the running example first. The name or the formula comes afterwards, as a label for what the viewer just saw.
- Whenever a number is said, it is one of the running example's actual values.
- A result that has a proof in the source gets the one reason it is believable, in terms of the picture. No lemma-by-lemma steps, no "suppose not", no theorem numbers.
- Accuracy beats neatness. Every sentence must be true as stated, for the running example and in general, and must agree with the source. When a tidy claim holds only in a special case, say the weaker thing that is always true.
- Opening: {opening_rule}
- Ending: the payoff act says the answer once. Its final beat answers the central question in one or two sentences and points back at the opening picture. Stop there. No second or third restatement, no summary list, no "in this video", no sign-off.

"visual_hint" (one or two sentences per beat, read by the animators)
- Describe what CHANGES on screen during the beat, using the motif names: what moves, grows, rotates, morphs, lights up or fades.
- It must be drawable with dots, arrows, lines, curves, axes, grids, simple shapes and short labels. Never ask for photographs, 3D scenes, people, hands or real footage. Draw an image as a small grid of gray squares, a dataset as a cloud of dots, a machine as labeled boxes with arrows.
- On-screen words are labels of at most four words, or one formula. Never ask for a sentence, a title or a caption on screen.
- Inside an act the same motif stays on screen from beat to beat: change it, do not replace it.

Other tags
- "introduces_concepts": lowercase underscored labels for ideas first taught in this beat. A concept is introduced once in the whole script.
- "references_prior_beats": ids of earlier beats whose picture this beat builds on.
- "source_segment_id": the one segment this beat takes most of its facts from, or "" for a beat that needs none.

{TYPED_BEAT_PROMPT_RULES}

{SPOKEN_STYLE_RULES}
Respond with one JSON object in exactly this shape (no other text):
{{
  "title": "...",
  "analogies": [
    {{"analogy_id": "analogy_short_name", "label": "what the analogy is", "target_concepts": ["concept_label"], "mapping": {{"analogy term": "concept"}}}}
  ],
  "beats": [
    {{
      "beat_id": "beat_001",
      "act_id": "act_01",
      "narration": "spoken text",
      "visual_hint": "what changes on screen during this beat",
      "introduces_concepts": ["concept_label"],
      "references_prior_beats": [],
      "source_segment_id": "seg_001",
      "role": "core",
      "depth": 0,
      "analogy_id": null
    }}
  ]
}}
"""


_SNAKE_CASE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b")


def spoken_plain(narration: str) -> str:
    """Narration with any snake_case name (a leaked motif or object id) turned into words.

    A text-to-speech voice reads ``best_fit_line`` as "best underscore fit
    underscore line"; the writer is told never to do this, and this makes sure.
    """
    return _SNAKE_CASE.sub(lambda m: m.group(1).replace("_", " "), narration or "")


def draft_listing(beats: Iterable[Any]) -> str:
    """A draft's beats as ``[act_id] narration`` lines, for a rewrite prompt."""
    return "\n".join(
        f"[{getattr(beat, 'act_id', '') or 'no act'}] {beat.narration.strip()}" for beat in beats
    )


def script_length_feedback(plan: Any, beats: Iterable[Any], word_budget: int) -> str:
    """Reviewer point for a planned script that came in under its length.

    Names each act that is more than 15 percent short of its planned words,
    so the rewrite adds material where it is missing. Returns "" when the
    script reaches ``SCRIPT_FLOOR_SHARE`` of the budget.
    """
    beats = list(beats)
    total = sum(len(beat.narration.split()) for beat in beats)
    floor_words = round(word_budget * SCRIPT_FLOOR_SHARE)
    if total >= floor_words:
        return ""
    by_act: Dict[str, int] = {}
    for beat in beats:
        act_id = getattr(beat, "act_id", "") or ""
        by_act[act_id] = by_act.get(act_id, 0) + len(beat.narration.split())
    short = [
        f"{act.act_id} has {by_act.get(act.act_id, 0)} words and needs about {act.words}"
        for act in plan.acts
        if by_act.get(act.act_id, 0) < act.words * 0.85
    ]
    return (
        f"The script is {total} words. It must be between {floor_words} and {word_budget}: the "
        "video has a set length. "
        + ("Short acts: " + "; ".join(short) + ". " if short else "")
        + "Make up the difference by adding the steps the viewer needs (one more thing that "
        "happens to the running example, shown and said), in new beats. Do not pad sentences "
        "and do not repeat anything."
    )


# ── 3. presenter bridges, written from the plan ─────────────────────────


def bridge_slot_line(slot: int, kind: str, before: str, after: str, next_question: str = "") -> str:
    """One slot of the bridge listing. ``next_question`` is the viewer
    question of the act that starts right after this bridge ("" if unknown)."""
    line = (
        f"- slot {slot} ({kind}): BEFORE: {before or '(video start)'} | "
        f"AFTER: {after or '(video end)'}"
    )
    if next_question:
        line += f" | QUESTION THE NEXT PART ANSWERS: {next_question}"
    return line


def bridge_prompt(slot_lines: List[str], plan: Any) -> str:
    """Presenter bridges for a planned video. Each bridge has one job and no content of its own."""
    return (
        "The professor appears on camera between animated scenes of a short explainer video. "
        "Write what they say in each slot below, in the first person.\n\n"
        f"The video's central question: {plan.central_question}\n"
        f"The hook: {plan.hook}\n"
        f"The takeaway: {plan.takeaway}\n\n"
        "What each kind of slot is for:\n"
        "- opening: say the hook to camera, in your own words, and end on the central question "
        "without answering it. At most 35 words. No greeting, no \"today we will\", no list of topics.\n"
        "- transition: one sentence that makes the viewer want the next part: what is still "
        "unexplained, or what is at stake. At most 20 words. The next scene asks its own question "
        "(shown as QUESTION THE NEXT PART ANSWERS), so do not ask that question or reword it, and "
        "do not summarize what came before.\n"
        "- why: say in one sentence why the next step is worth the effort. At most 25 words.\n"
        "- recap: say the takeaway once, plainly, then one sentence on what it lets the viewer do "
        "or notice. At most 35 words. No list of topics covered.\n\n"
        "Rules for every slot:\n"
        "- A bridge raises a question or lands a conclusion. It never explains: no formula, no "
        "definition, no analogy, no step of an argument.\n"
        "- The AFTER text is spoken right after your bridge, so never say what it says.\n"
        "- Never restate a fact, number or conclusion the video has already said (in the hook or any "
        "earlier scene). Point forward to what is still open instead.\n"
        "- Sound like a person talking: contractions, short sentences, math said in words. No stock "
        "phrases (\"let's dive in\", \"here's the key\", \"building blocks\", \"now that we have\", "
        "\"before we get into the details\"), no em dashes, no exclamation marks.\n\n"
        + "\n".join(slot_lines)
        + "\n\nRespond in JSON as {\"bridges\": [{\"slot\": 0, \"text\": \"...\"}]}"
    )


def plan_summary_for_review(plan: Any) -> Dict[str, Any]:
    """The plan fields a reviewer needs, as plain data."""
    return {
        "central_question": plan.central_question,
        "hook": plan.hook,
        "promise": plan.promise,
        "takeaway": plan.takeaway,
        "running_example": plan.running_example.description,
        "motifs": [m.motif_id for m in plan.motifs],
        "acts": [
            {"act_id": a.act_id, "kind": a.kind, "viewer_question": a.viewer_question,
             "key_idea": a.key_idea, "words": a.words}
            for a in plan.acts
        ],
    }
