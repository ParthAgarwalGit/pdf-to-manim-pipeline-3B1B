"""Model-facing instruction text for the V2 pipeline.

Only words live here: functions take plain values and return finished
instruction strings. The stages in planning.py, generation.py, tts.py and
presenter_stage.py decide when to call a model and what data to append.

Bump the matching prompt-version constant in the calling module whenever the
text a function returns changes, so cached responses are not reused.
"""

from __future__ import annotations

from typing import Optional

# The one prebuilt Gemini voice shared by narration (Gemini TTS) and the
# on-camera presenter (Gemini Omni speech_config), so both sound like one person.
DEFAULT_VOICE = "Sulafat"

# Phrases that make narration sound machine-written. Shared by the narration
# prompt and the narration review.
STOCK_PHRASES = (
    "let's dive in", "let's take a look", "here's the key", "here's the thing",
    "here's the twist", "there's a catch", "it turns out", "it's important to note",
    "as we can see", "in this video", "simply put", "basically", "essentially",
    "in essence", "delve", "crucial", "fascinating", "building blocks",
    "here's where it gets interesting", "this is where", "the key insight", "it's worth noting",
    "at its core", "plays a crucial role", "elegant", "seamless", "unlock", "game-changer",
    "in conclusion", "to summarize", "in summary", "let's break it down", "deep dive",
)


# ── lesson plan ─────────────────────────────────────────────────────────


# Hook word limits: spoken by an on-camera presenter, or by the narrator over
# the first picture. planning.hook_problems() enforces them.
# 34 words fit one presenter clip (about 13 s): a longer opening is split into
# separately generated clips with a visible jump between them.
# Spoken by the presenter in one clip: longer hooks ran out of clip time.
HOOK_MAX_WORDS_PRESENTER = 30
HOOK_MAX_WORDS_NARRATOR = 45

HOOK_CHAIN = """Build the hook as a chain of three or four short sentences, each one following from the sentence before it:
     (a) the situation, in the running example's terms: who or what we are looking at, what we know, and what we want to find out;
     (b) the obvious first attempt, or what anyone would expect to happen;
     (c) the snag: what goes wrong with that attempt, and why, stated so that it is true for the running example's actual values;
     (d) the central question in at most 15 words, left unanswered.
   Introduce every person, object, quantity and number before you use it. Never state a conclusion the listener has no reason yet to believe.
   The hook poses the problem; it never gives away what the video works out. Leave every count, value and result that a scene derives to that scene (say "the laws we have are not enough", not "sixteen unknowns but only five equations"). Exactly one question mark, at the end. No dashes, semicolons or parentheses.
   Bad, every sentence jumps: "A café has one barista. A single line breaks the queue rules. How long is the wait?" The listener never learns what is measured, what was tried or why anything breaks.
   Good, each sentence needs the one before: "A café gets a customer every two minutes, and each drink takes two minutes. You'd expect no line at all. Yet a line forms and keeps growing. Why?\""""


# Video modes. "brief" is the short concept visualizer (about five minutes, an
# overview of the module); "deep" is the in-depth lecture that teaches the
# whole document, derivations and worked examples included.
MODES = ("brief", "deep")
MODE_LABELS = {
    "brief": "Brief concept visualizer",
    "deep": "In-depth lecture",
}
MODE_DESCRIPTIONS = {
    "brief": "a short overview: every main concept of the document, shown visually",
    "deep": "up to about 25 minutes: every important concept, derivation and worked example",
}


def lesson_instructions(
    scene_guidance: str,
    presenter_opens: bool,
    mode: str = "brief",
    audience: str = "undergraduate",
) -> str:
    """What the lesson planner is asked to do, before the facts JSON is appended.

    ``scene_guidance`` is planning.py's scene-count sentence(s). ``presenter_opens``:
    an on-camera presenter speaks the hook before the first animation. ``mode``
    is "brief" (one question, shortest path) or "deep" (the whole document).
    """
    deep = mode == "deep"
    if presenter_opens:
        hook_rule = (
            'A presenter speaks "hook" on camera before any animation, so it must make complete '
            f"sense with no picture: first person, at most {HOOK_MAX_WORDS_PRESENTER} words. "
            f"{HOOK_CHAIN}\n   The hook is not a scene. The first scene opens directly on the "
            "running example and shows the snag happening."
        )
    else:
        hook_rule = (
            '"hook" is the first thing the narrator says, over the first scene\'s opening '
            f"picture: at most {HOOK_MAX_WORDS_NARRATOR} words, no greeting, no topic "
            f"announcement. {HOOK_CHAIN}"
        )
    if deep:
        intro = (
            "You are the story editor for an in-depth animated lecture, the kind made by the best "
            "visual math and science channels. It teaches a student meeting this material for the "
            "first time everything important in the document whose source facts follow: every key "
            "concept, every derivation that matters, the intuition for why each result has the form "
            "it has, and how to use it to solve problems. It may run up to about 25 minutes, but "
            "every minute must teach something. A narration writer and Manim animators work from "
            "your plan, so every field must be concrete enough to act on."
        )
        question_scope = (
            "The question the whole lecture answers, in a curious student's words; each chapter "
            "answers one part of it."
        )
        examples_rule = (
            "Worked examples may use further small examples: the document's own examples and "
            "exercises first, otherwise a small instance of the document's own method with simple "
            "numbers you choose (an illustration of its method, not a new fact)."
        )
        motif_count = "Two to four"
        scene_rules = f"""   - Group the scenes into 3 to 7 chapters (field "chapter": a short title, the same for every scene of a chapter). Each chapter answers one question that the previous chapter leaves open, and its first scene's viewer_question is that question.
   - Give every scene a "role": "motivate" (make a problem felt), "intuition" (why the result should look the way it does, shown with the running example before any algebra), "derive" (the derivation the document gives, one scene for every few steps, with each step's reason in the plan), "apply" (a worked example used the way the document's exercises would use the result) or "connect" (how a result ties to an earlier one, or where it stops working). Every important result gets intuition, then derive when the document derives it, then apply. A plain definition needs no derive scene.
   - A derive scene's learning_goal names the step it takes and its visual_objective names the equation it starts from and the one it ends with. An apply scene states its problem and numbers in "example".
   - Cover every substantive concept, every definition the document goes on to use, every derivation, algorithm and worked example. Prerequisites the document assumes are not scenes: a {audience} student already knows them, so the narration mentions one in a sentence where it is needed. Spend the time on what the document adds, not on basics.
   - Every later scene follows from the one before it: it answers what the previous scene left open ("viewer_question") and adds one new idea ("learning_goal", one plain sentence).
   - Show before you name: an idea first appears as something visibly happening to the running example; its name and formula come afterwards, as a label for what was just seen.
   - "visual_objective" says what the viewer WATCHES CHANGE in that scene, in terms of the motifs or the equation chain: what moves, stretches, splits, lines up, cancels or collapses. Never what text appears.
   - "new_visual_fact" is the one thing the picture shows that no earlier scene showed; "do_not_repeat" lists earlier constructions this scene must not rebuild.
   - The last scene returns to the first scene's picture and answers central_question with everything the lecture built; "takeaway" is that answer in one sentence.
   - Every scene cites the source_fact_ids its facts come from. Do not invent facts or equations; worked-example numbers follow the rule in step 2.
   - Accuracy beats neatness. Every learning_goal must be true as stated, for the running example and in general.
   - Derivations and proofs the document gives are planned step by step across derive scenes, at the depth the document gives them; only routine algebra is skipped.
   - Set delivery_mode to manim for every scene."""
        coverage_rule = (
            '5. COVERAGE ("coverage"). One entry per source page. Every page that teaches anything '
            'is "covered". Only pages with no teaching content (title, outline, references, '
            'administration) may be "omitted", each with a one-sentence reason.'
        )
        checks = """- Does every important result have an intuition scene, a derive scene when the document derives it, and an apply scene?
- Could a student who watched only this lecture follow the document's derivations and solve its exercises?
- Is any scene spent on a prerequisite the audience already knows? Cut it."""
    else:
        intro = (
            "You are the story editor for a short animated explainer, the kind made by the best "
            "visual math and science channels. The source facts of a document follow. Your job is "
            "not to summarize them. Decide the ONE question this video answers and the shortest "
            'visual path from "never seen this" to "I see why". A narration writer and Manim '
            "animators work from your plan, so every field must be concrete enough to act on."
        )
        question_scope = (
            "The question a curious viewer would actually ask, in their words, that the document's "
            "main result answers."
        )
        examples_rule = ""
        motif_count = "Two to four"
        scene_rules = """   - Every later scene answers exactly one question the previous scene left the viewer with ("viewer_question") and adds exactly one new idea ("learning_goal", one plain sentence).
   - Show before you name: an idea first appears as something visibly happening to the running example; its name and formula come afterwards, as a label for what was just seen.
   - Put each idea where the viewer first needs it. Cut definitions the argument never uses.
   - "visual_objective" says what the viewer WATCHES CHANGE in that scene, in terms of the motifs: what moves, stretches, rotates, splits, lines up or collapses. Never what text appears.
   - "new_visual_fact" is the one thing the picture shows that no earlier scene showed; "do_not_repeat" lists earlier constructions this scene must not rebuild.
   - The last scene returns to the first scene's picture and answers central_question; "takeaway" is that answer in one sentence.
   - Every scene cites the source_fact_ids its facts come from. Do not invent facts, examples or equations.
   - Accuracy beats neatness. Every learning_goal must be true as stated, for the running example and in general. When a tidy connection holds only in a special case, plan the weaker statement that is always true.
   - Formal proofs: unless the instructions below ask for proofs, keep only the one idea that makes a result believable, shown as a picture inside the scene that needs it. Never plan a scene whose job is to walk through the steps of a proof.
   - Set delivery_mode to manim for every scene."""
        coverage_rule = (
            '5. COVERAGE ("coverage"). One entry per source page: "covered" (a scene explains it), '
            '"compressed" (a scene states its conclusion without the working), "deferred" (named '
            'in passing as beyond this video) or "omitted", with a one-sentence reason for anything '
            "not covered. Every main concept the document teaches (its definitions, conditions, "
            "algorithms and results) appears in some scene, if only in a sentence: this video is "
            "an overview of the whole document, not of one part of it. Leave out only side "
            "remarks, administration and material off the main line, and say why here. A "
            "full-coverage request in the scene instructions above overrides this: then every "
            "substantive page is covered."
        )
        checks = ""
    return f"""{intro}

{scene_guidance}

HOW TO PLAN

1. ONE CENTRAL QUESTION ("central_question"). {question_scope} One sentence, at most 25 words, answerable with a picture. A topic is not a question: not "What is X?" or "An overview of X", but a question with tension in it, where something looks impossible, wasteful or surprising until the document's main idea explains it.

2. RUNNING EXAMPLE ("running_example"). One small concrete case, with actual numbers taken from the facts whenever the document has one, that the whole video keeps returning to. It must be drawable on a flat screen: two dimensions, a handful of points, single-digit numbers. Give its values in "values" (name to value), plus the symbols the video will show for its main objects, written exactly as the document writes them. {examples_rule}

3. MOTIFS ("motifs"). {motif_count} recurring on-screen objects built from the running example, each with a fixed color and a fixed meaning; the viewer learns the picture once. Colors: BLUE_B, TEAL_C, GREEN_C, YELLOW_C, GOLD_C, RED_C, MAROON_C, PURPLE_C, WHITE, GRAY_B. Drawn with dots, arrows, lines, curves, grids, circles, rectangles and short labels.

4. SCENES, in viewing order, not document order.
   - The first scene opens on the running example doing something that raises the central question: a puzzle, a surprising outcome, or a task that looks hard. No definitions, no history. {hook_rule}
{scene_rules}

{coverage_rule}

CHECK BEFORE YOU ANSWER
- Read the hook aloud as someone who has never seen the document. Is every person, quantity and number introduced before it is used? Does each sentence follow from the one before? Is the snag true for the running example's values (check the arithmetic)?
- Could a viewer say what the central question is after the first twenty seconds?
- Does every scene's visual_objective use the running example, a motif or the equation chain?
- Read the learning_goals in order: is it one argument in which each step needs the one before it?
- Does the last scene answer central_question?
- Is every page in "coverage" once?
{checks}"""


def hook_check_instructions() -> str:
    """Asks whether each sentence of the planned hook follows; JSON payload follows."""
    return """You check the opening lines of a short explainer video for a viewer who has never seen the topic and hears each sentence once. The hook, the central question it leads to and the running example (with its actual values) are given below.

Read the hook one sentence at a time. A sentence fails when:
- it mentions a person, object, quantity, number or symbol that no earlier sentence (and no part of the same sentence) has introduced;
- it does not follow from the sentence before it: the listener could not say why it comes next (because, so, but);
- it states a conclusion before the setup that makes it believable;
- a claim in it is false for the running example's actual values. Check the arithmetic;
- it gives away a count, value or result that the video should let the viewer work out (the hook poses the problem; the scenes answer it).

List every failing sentence exactly as written, with what the listener is missing or why it is false. Do not rewrite anything.

Return only JSON: {"breaks": [{"sentence": "...", "problem": "..."}]}, with "breaks": [] when every sentence passes."""


def hook_rewrite_instructions(presenter_opens: bool, problems: list[str]) -> str:
    """Asks for a new hook when the planned one fails its checks; JSON payload follows.

    ``problems`` are plain sentences, from planning.hook_problems() or from the
    script reviewer's flow breaks in the hook.
    """
    if presenter_opens:
        spoken = (
            "a presenter says it on camera before any animation, so it must make complete sense "
            f"with no picture. First person, at most {HOOK_MAX_WORDS_PRESENTER} words."
        )
    else:
        spoken = (
            "the narrator says it over the first scene's opening picture. No greeting, no topic "
            f"announcement, at most {HOOK_MAX_WORDS_NARRATOR} words."
        )
    listed = "\n".join(f"- {problem}" for problem in problems)
    return f"""Rewrite the opening hook of a short animated explainer: {spoken}

WHAT IS WRONG WITH THE CURRENT HOOK
{listed}

HOW TO WRITE IT
   {HOOK_CHAIN}

Use only the running example, the central question and the first scene given below. Do not bring in a new example, fact or number, and do not answer the question. Plain spoken words: say symbols and numbers the way you would out loud.

Return only JSON: {{"hook": "the new hook"}}"""


# ── narration ───────────────────────────────────────────────────────────


# Longest sentence the narration may say; script_review.voice_problems() checks it.
MAX_SPOKEN_SENTENCE_WORDS = 30

SPOKEN_STYLE = f"""SPOKEN NARRATION (a voice reads it aloud; it must sound like one person thinking out loud with a student, not an AI summary or a textbook):
- Talk about the ideas, not the drawing. Call things by what they are or do ("the sender", "P two's tally", "the slope"), never by color or shape ("the red double-circled node", "the gold arrow labeled zero"). Say a color only when two things on screen could not be told apart otherwise, and then once.
- At most one pointer per scene ("look at", "notice", "watch"). The animation already shows where the action is, so say what happens, not where to look. Never a stage direction as a sentence ("Compare the two cases side by side.").
- Let each thought lead into the next, the way people reason out loud: "so", "but", "which means", "and that's why", "now suppose". A beat's first sentence picks up where the last beat ended.
- Vary the rhythm: mostly short sentences, the occasional longer one, never more than {MAX_SPOKEN_SENTENCE_WORDS} words. One idea per sentence. Plain words and contractions (it's, we'll, that's).
- Bring in a term the way a person would: what the thing does, in plain words, first; its name after ("...a message that also lists who vouched for it. Call that a star message.").
- Never sum a scene up with a definition-shaped sentence ("In Byzantine agreement, all processors must...", "By relaying values, the processors achieve...", "This is called..."). Land the point as a plain consequence of what was just seen ("So one liar can't split them any more.").
- Never use stock phrases: {", ".join(f'"{p}"' for p in STOCK_PHRASES)}. No exclamation marks, semicolons, parentheses, em dashes, lists or headings. At most one question per scene.
- Say math the way you would out loud: "A equals U times D times V transpose", "sigma one", "the length of A v". Never put symbols such as ^T, _k, ||x|| or LaTeX in beat text.
- Names with underscores (motif ids, scene ids) never appear in beat text."""


def narration_instructions(
    pace_guidance: str,
    target_wpm: int,
    presenter_opens: bool,
    mode: str = "brief",
    audience: str = "undergraduate",
) -> str:
    """What the narration writer is asked to do, before the payload JSON is appended.

    ``pace_guidance`` is planning.py's pace sentence(s). The payload carries
    "lesson_spine" (central_question, hook, takeaway, running_example, motifs)
    and each scene's viewer_question (and, in deep mode, chapter and role).
    """
    deep = mode == "deep"
    if presenter_opens:
        opening = (
            'An on-camera presenter has already said the hook (lesson_spine.hook). Do not repeat '
            "it. The first scene's first beat starts on the picture and goes straight to work on "
            "the question."
        )
    else:
        opening = (
            "The first scene's first beat is the hook (lesson_spine.hook), said over the opening "
            "picture nearly as written. No greeting, no topic announcement, no definitions."
        )
    if deep:
        proofs = (
            "- Derivations are worked step by step: each beat takes one step, says what is done and "
            'why it is allowed ("we can divide by p because p is never zero here"), and the picture '
            'shows that same step. Skip only routine algebra, saying so in a phrase ("after '
            'tidying up"). A worked example says the problem, the plan, each step, and then checks '
            "that the answer makes sense."
        )
        ending = (
            "- Ending: the last scene answers central_question with what the lecture built, then "
            "says in one or two sentences what the viewer can now do with it. No list, no sign-off."
        )
        beats = (
            "- A beat is one visual moment: one to four sentences, 25 to 60 words. One derivation "
            "step or one worked-example step is one beat. When the picture has to change twice, "
            "that is two beats."
        )
        deep_block = f"""
IN-DEPTH LECTURE (this video teaches the whole document to a student meeting it for the first time; it may run up to about 25 minutes)
- Each scene has a role (scenes_to_write[].role). "motivate" makes the problem felt. "intuition" shows why the result should look the way it does, on the running example, before any algebra. "derive" works the derivation step by step. "apply" solves a worked example. "connect" ties a result to an earlier one or shows where it stops working.
- A chapter's first scene (a new scenes_to_write[].chapter) orients the viewer in one or two sentences: what we now know and what is still open. It is not a recap.
- Assume what a {audience} student in this subject already knows. A prerequisite gets one sentence where it is needed, never a scene's worth, unless the document itself teaches it.
- Depth comes from reasons, not from extra words: every formula is explained by where it comes from and what each part of it does, and every method by when to reach for it and how to use it.
"""
    else:
        proofs = (
            "- A result with a proof in the facts gets the one reason it is believable, in terms "
            'of the picture. No lemma-by-lemma steps, no "suppose not", no theorem numbers.'
        )
        ending = (
            "- Ending: the last scene says the answer to central_question once, points back at "
            "the opening picture and stops. No second restatement, no summary list, no sign-off."
        )
        beats = (
            "- A beat is one visual moment: one to three sentences, 20 to 45 words. When the "
            "picture has to change twice, that is two beats."
        )
        deep_block = ""
    return f"""Write the requested consecutive section of one spoken lesson. One person speaks it over animation. The lesson plan and its spine (lesson_spine) are settled: the central question, the running example, the motifs and the scene order are not yours to change.

{pace_guidance}
{deep_block}
SHAPE
- A scene starts where the previous one stopped. Open on what the viewer is wondering now: a consequence of the last scene ("So the receivers can't simply trust the sender."), a complication ("But that only works when..."), a fresh case, or a question. Open with a question in at most every other scene.
- A scene ends on its point, said once in plain words as a consequence of what was just seen, or on the question it hands to the next scene. Never on a recited definition.
- Show before naming: describe what happens to the running example first; the name or the formula comes afterwards, as a label for what the viewer just saw.
- Whenever a number is said, it is one of the running example's actual values, and a sentence says at most two numbers. Never read out a vector, a list of iterates, a formula with more than three terms or a number with more than three significant figures: round it ("about seventy-eight percent") or say what it does ("the slope settles near one point two").
{proofs}
- Opening: {opening}
{ending}
- The first beat of every other scene follows straight on from the previous scene's last beat (continuity.previous_scene); never recap it.

FLOW (the listener hears each sentence once and cannot scroll back)
- Each sentence follows from the one before it. When the link is not obvious, say it: "because", "so", "but", "which means".
- Introduce before you use: never mention a person, object, quantity or symbol ("the new student", "beta one", "the slope") before the narration, or the picture it describes, has said what it is.
- Set up before you claim: say what happens first, then what it means. A claim about the running example must be true for its actual values; check the arithmetic before you write it.
- Do not skip a step the viewer needs to follow the argument, and do not take two steps in one sentence.

ECONOMY (this is what separates a script from a transcript of a lecture)
- Say each thing once. Every sentence moves the thought forward.
- No announcements ("now we will see"), no recaps of what was just shown.
- Let the picture carry the detail: do not read out coordinates, matrix entries or every term of a formula that is on screen. Say what matters about them.
- Prefer the short concrete word: "stretch", not "apply a scaling transformation".
- Accuracy beats neatness: every sentence must be true as stated and agree with the cited facts. When a tidy claim holds only in a special case, say the weaker thing that is always true.

BEATS
{beats}
- visual_action says what CHANGES on screen during the beat, using the motif names: what moves, grows, rotates, morphs, lights up or fades. It must be drawable with dots, arrows, lines, curves, axes, grids, simple shapes, labels of at most four words and, for a derivation or worked example, the equation or working being transformed. Never ask for a sentence, title or caption on screen, and never for photographs or 3D scenes.
- Inside a scene the same motif stays on screen from beat to beat: change it, do not replace it.

{SPOKEN_STYLE}

FORMAT
Return exactly the required_scene_ids once each and in the supplied order. Populate role, pause_after_seconds (normally 0.25-0.55, or 0.6-0.9 for reflection and scene endings), visual_action, continuity_note, transition_in and transition_out. Every beat cites only supplied source_fact_ids. Preserve mathematical notation in the facts you cite, but speak it in words. Set target_words_per_minute to {target_wpm}. Timing fields must be zero; the pipeline computes them after the audio is recorded."""


# ── targeted flow fix ───────────────────────────────────────────────────


# How many words a fixed beat may add: room to say what one term means.
BEAT_FIX_MAX_EXTRA_WORDS = 15


def beat_fix_instructions(problems: list[str]) -> str:
    """Asks for one beat rewritten so its flagged sentences follow; JSON payload follows.

    ``problems`` are plain sentences built from the script reviewer's flow
    breaks in that beat. The payload carries the beat (beat_to_fix), every
    beat of its scene (scene_beats), the previous scene's last beat and the
    facts the beat cites.
    """
    listed = "\n".join(f"- {problem}" for problem in problems)
    return f"""Fix one beat (beat_to_fix) of a spoken lesson script. The listener hears each sentence once and cannot scroll back. A reviewer found sentences in this beat that the listener cannot follow, or that sound written or machine-made rather than spoken:

WHAT IS WRONG
{listed}

HOW TO FIX IT
- Change only what these problems need. Every other sentence of the beat stays as written, and every other beat stays as it is.
- When a term, symbol or acronym is used before anyone has said what it is, keep it and introduce it in plain words where it first comes up: what it is or what it measures, in terms of what the viewer sees. Do not just delete the idea.
- When a sentence does not follow from the one before it, add the missing link ("because", "so", "but") or the missing step.
- When a sentence describes the drawing (colors, shapes, "the arrow labeled one") or tells the viewer where to look, say what the thing is or what happens instead. When it recites a definition or summary, say the point as a plain consequence. When it is too long, split it. Keep its meaning.
- Bring in no other new term, symbol, acronym, number or fact. Use only the facts this beat cites (source_facts).
- The beat comes after the beats before it in scene_beats (or, as its scene's first beat, after previous_scene_last_beat) and leads into the beats after it: do not repeat what they say, and keep what the later beats rely on.
- The fixed beat is at most {BEAT_FIX_MAX_EXTRA_WORDS} words longer than beat_to_fix.

{SPOKEN_STYLE}

Return only JSON: {{"text": "the whole fixed beat"}}"""


# ── voice ───────────────────────────────────────────────────────────────


def tts_style(audience: Optional[str] = None) -> str:
    """How the narration is delivered, sent to Gemini TTS as speech_metadata style.

    It never goes in the spoken text: the TTS model reads its whole text
    input aloud, so a direction placed there is heard in the video.
    """
    listener = f"a {audience} student" if audience else "one student"
    return (
        f"A warm, curious teacher explaining an idea she finds genuinely interesting to "
        f"{listener} sitting beside her: natural rise and fall in pitch, a small lift on the "
        "surprising or important word in each sentence, a brief pause at each comma. Unhurried "
        "but never slow, never flat, never announcer-like."
    )


# ── code generation ─────────────────────────────────────────────────────


CODEGEN_STYLE_RULES = """
Composition and text:
- Fill the frame. The main visual is the largest thing on screen: a coordinate system that is the subject of the scene uses x_length 9 to 11 and y_length 5.5 to 6.5 (6 to 7.5 by 4.5 to 5.5 when a formula shares the frame). A small diagram in a sea of black is a defect.
- Keep one evolving picture. The main visual appears within the first two seconds and stays; later beats change it. Do not clear the screen mid-scene to start a different diagram. Dim what is not being discussed instead of deleting it.
- The narration is heard, so the screen never repeats it: no sentences, titles, headings or captions. Text() is for labels of at most four words beside what they name; mathematics goes in MathTex, built term by term and colored to match the objects its terms describe. At most one new formula per scene, except in a derivation (brief.role "derive") or a worked example (brief.role "apply"), whose rules are below.

Layout (the frame is 14.2 by 8 units; everything stays inside x from -6.6 to 6.6 and y from -3.6 to 3.6):
- Decide the regions first: the main picture in the centre or on one side, formulas in a column on the other side. A formula placed beside a diagram gets .scale_to_fit_width() of the space actually left between the diagram and the frame edge when it would not fit.
- A label goes beside what it names, with next_to(target, direction, buff=0.2) on the side that has room, never on top of it. A label for a line or arrow sits beside the line's midpoint, offset perpendicular to the line, never centred on it.
- A label that belongs inside a shape (a node's name) is built with it as VGroup(shape, label) so they move together, and lines and arrows end at the shape's border (buff=0.1 or more), never at its centre.
- Before new text appears where text already is, FadeOut or Transform the old text. Two pieces of text never share space, and a line never runs through text.
- About eight things on screen at once at most; FadeOut what the narration has moved past. Labels use font_size 24 to 32, formulas 30 to 40.
- Plan the whole path of anything that moves, not only where it ends. A dot, arrow, label or group moved with .animate, MoveAlongPath or an updater never passes over other text on the way; move or fade what is in its path first. A label of a moving object follows it (add_updater(lambda m: m.next_to(target, direction, buff=0.2)) or always_redraw), and stays inside the frame at every position the object reaches.
- When a shape grows, a curve extends or a group is rescaled, the labels beside it move with it (put them in the same VGroup or give them updaters), so the grown shape never covers them and they never leave the frame.

Derivations and worked examples:
- A derivation (brief.role "derive") is one chain of equations in one place. Each step turns the previous line into the next with TransformMatchingTex or ReplacementTransform, coloring the term that changes and briefly showing why it changes (an arrow from the reason to the term, or the reason as a short label of at most four words beside it). At most three lines are visible; when a fourth is needed, fade the oldest. The result of the chain ends inside a SurroundingRectangle.
- A worked example (brief.role "apply") states the problem as at most two short lines of MathTex at the top, works it below one step per beat in the same column, and boxes the answer. When the example has a picture (a graph, a network, a diagram), the picture changes with each step.
- Axis titles sit fully inside the frame. A value read off an axis is placed clear of that axis's tick numbers.
- Draw each mark once: an existing bar, dot or line is moved or resized with .animate or a ValueTracker, never drawn again on top of itself.
- Pace: Transform and moves take at least 1.0 s, Create and Write at least 0.6 s. Two or three unhurried changes per beat teach more than six quick ones.
- Keep the picture alive while the narrator talks: never hold the frame unchanged for more than about five seconds. When a beat runs long, spread its changes across it, or use the extra time for a small motion that points at what is being said (a dot travelling along the curve, a highlight sweeping the term, a ValueTracker easing a value).
- When "visual_world" is given, draw its running example with exactly those values and give each motif its stated color and meaning, so every scene of the video shares one look."""


REPAIR_RECIPES = """
How to fix each kind of finding (object names in findings are variable names in the code; a name like nodes.2.1 is part 1 of item 2 of nodes):
- TEXT_FRAME_COLLISION or OFFSCREEN_OBJECT: it crosses the frame edge. Keep it beside what it labels but make it fit: put it on the side with room, scale_to_fit_width() to the space left, or move the whole picture inward. Everything stays inside x from -6.6 to 6.6 and y from -3.6 to 3.6.
- TEXT_TEXT_OVERLAP: two pieces of text share space. FadeOut or Transform the older one before the newer one appears, or place the newer one with next_to(..., buff=0.25) on a free side.
- TEXT_GEOMETRY_OVERLAP: a label sits on a line, arrow or shape. Move the label beside it: for a line, beside its midpoint and offset perpendicular to it; for a shape, next_to the shape. A name that belongs inside a node is grouped with the node, and lines end at the node's border.
- VISUAL_DENSITY_HIGH: too much on screen. FadeOut what the narration has moved past.
- TEXT_TOO_SMALL: labels use font_size 24 or more, formulas 30 or more.
- BEAT_LATE or SCENE_OVERRUN: shorten the run_time of that beat's animations.
- VISUAL_TEXT_OVERLAP: seen in the rendered video at the time in the message, often while something moves; the first images show it. Find the objects from what their text says. Move the label beside its target on a free side, fade the older text before the newer appears, keep a line's end at a node's border, or make a moving object's label follow it with an updater so nothing crosses it at any moment.
- VISUAL_CUT_OFF: seen in the rendered video: part of an object leaves the frame at that time, often at the end of a move or as a group grows. Keep every position it reaches inside x from -6.6 to 6.6 and y from -3.6 to 3.6: shift or scale the group, or shorten the move.
- VISUAL_CROWDED: objects pile up at that time. Spread them out or fade what the narration has finished with.
Fix every listed finding, starting with the first; change nothing else."""


# ── visual QA ───────────────────────────────────────────────────────────

# Defect kinds the visual check may report.
VISUAL_QA_KINDS = ("text_overlap", "line_through_text", "cut_off", "pile_up")


def visual_qa_instructions(timestamps: list[float]) -> str:
    """Asks a vision model to find layout defects a viewer would notice in a scene's frames."""
    listed = ", ".join(f"image {index} at {seconds:.1f} s" for index, seconds in enumerate(timestamps, start=1))
    return f"""You check frames from one scene of an animated math explainer for layout defects a viewer would notice. The images are in order: {listed}. Frames taken during an animation show objects mid-move; check those too.

Report a defect only when it is clearly visible:
- text_overlap: two pieces of text drawn on top of each other, so either is harder to read.
- line_through_text: a line, arrow, curve or shape edge drawn across the letters of a label or formula.
- cut_off: text or a shape partly outside the edge of the frame.
- pile_up: several objects stacked so close together that you cannot tell them apart.

Do not report: a label beside or inside the shape it names; text on a filled box made for it; an arrow that ends at the border of a node; objects that are merely near each other; text that is still being written or fading in or out; anything about style, color or design.

Report each problem once: list every image it appears in, give the one image that shows it most clearly ("best"), and say which objects in plain words (what the text says and where it is), so someone can find it in the code.

Return only JSON: {{"defects": [{{"kind": "one of {', '.join(VISUAL_QA_KINDS)}", "images": [3, 4], "best": 4, "what": "the label '1' on the arrow from P0 to P2 is crossed by that arrow"}}]}}, with "defects": [] when every frame is clean."""


def visual_qa_confirm_instructions(kind: str, what: str, frame_count: int = 1) -> str:
    """Asks whether one reported defect is really visible in any of the frames it was seen in."""
    if frame_count == 1:
        shown, look = "Here is one frame", "Look closely at the image. Is that exact defect clearly visible"
    else:
        shown = f"Here are {frame_count} frames, in order,"
        look = "Look closely at each image. Is that exact defect clearly visible in at least one of them"
    return f"""{shown} from an animated math explainer. A first look reported this layout defect ({kind.replace('_', ' ')}): "{what}"

{look}, so that a viewer would notice it? Answer no if the objects are only near each other, if the text is fully readable and nothing crosses it, if the described objects are not in the image, or if it is something still being drawn.

Return only JSON: {{"visible": true or false, "frame": the number of the image that shows it most clearly (1 if only one), "where": "where in the frame, in a few words, or empty"}}"""


# ── narration review ────────────────────────────────────────────────────

REVIEW_PASS_SCORE = 6.5
# A script also fails review when its logical_flow score is below this, or
# when the reviewer lists a sentence that does not follow from the last.
FLOW_PASS_SCORE = 6.0
