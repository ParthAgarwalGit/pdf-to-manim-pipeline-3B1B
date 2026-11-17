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
)


# ── lesson plan ─────────────────────────────────────────────────────────


# Hook word limits: spoken by an on-camera presenter, or by the narrator over
# the first picture. planning.hook_problems() enforces them.
HOOK_MAX_WORDS_PRESENTER = 40
HOOK_MAX_WORDS_NARRATOR = 45

HOOK_CHAIN = """Build the hook as a chain of three or four short sentences, each one following from the sentence before it:
     (a) the situation, in the running example's terms: who or what we are looking at, what we know, and what we want to find out;
     (b) the obvious first attempt, or what anyone would expect to happen;
     (c) the snag: what goes wrong with that attempt, and why, stated so that it is true for the running example's actual values;
     (d) the central question in at most 15 words, left unanswered.
   Introduce every person, object, quantity and number before you use it. Never state a conclusion the listener has no reason yet to believe. Exactly one question mark, at the end. No dashes, semicolons or parentheses.
   Bad, every sentence jumps: "A café has one barista. A single line breaks the queue rules. How long is the wait?" The listener never learns what is measured, what was tried or why anything breaks.
   Good, each sentence needs the one before: "A café gets a customer every two minutes, and each drink takes two minutes. They keep pace, so you'd expect no line. Yet a line forms and keeps growing. Why, when the barista is never too slow?\""""


def lesson_instructions(scene_guidance: str, presenter_opens: bool) -> str:
    """What the lesson planner is asked to do, before the facts JSON is appended.

    ``scene_guidance`` is planning.py's existing scene-count sentence(s)
    (full-coverage or short-document wording). ``presenter_opens``: an
    on-camera presenter speaks the hook before the first animation.
    """
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
    return f"""You are the story editor for a short animated explainer, the kind made by the best visual math and science channels. The source facts of a document follow. Your job is not to summarize them. Decide the ONE question this video answers and the shortest visual path from "never seen this" to "I see why". A narration writer and Manim animators work from your plan, so every field must be concrete enough to act on.

{scene_guidance}

HOW TO PLAN

1. ONE CENTRAL QUESTION ("central_question"). The question a curious viewer would actually ask, in their words, that the document's main result answers. One sentence, at most 25 words, answerable with a picture. A topic is not a question: not "What is X?" or "An overview of X", but a question with tension in it, where something looks impossible, wasteful or surprising until the document's main idea explains it.

2. RUNNING EXAMPLE ("running_example"). One small concrete case, with actual numbers taken from the facts whenever the document has one, that the whole video keeps returning to. It must be drawable on a flat screen: two dimensions, a handful of points, single-digit numbers. Give its values in "values" (name to value), plus the symbols the video will show for its main objects, written exactly as the document writes them.

3. MOTIFS ("motifs"). Two to four recurring on-screen objects built from the running example, each with a fixed color and a fixed meaning; the viewer learns the picture once. Colors: BLUE_B, TEAL_C, GREEN_C, YELLOW_C, GOLD_C, RED_C, MAROON_C, PURPLE_C, WHITE, GRAY_B. Drawn with dots, arrows, lines, curves, grids, circles, rectangles and short labels.

4. SCENES, in viewing order, not document order.
   - The first scene opens on the running example doing something that raises the central question: a puzzle, a surprising outcome, or a task that looks hard. No definitions, no history. {hook_rule}
   - Every later scene answers exactly one question the previous scene left the viewer with ("viewer_question") and adds exactly one new idea ("learning_goal", one plain sentence).
   - Show before you name: an idea first appears as something visibly happening to the running example; its name and formula come afterwards, as a label for what was just seen.
   - Put each idea where the viewer first needs it. Cut definitions the argument never uses.
   - "visual_objective" says what the viewer WATCHES CHANGE in that scene, in terms of the motifs: what moves, stretches, rotates, splits, lines up or collapses. Never what text appears.
   - "new_visual_fact" is the one thing the picture shows that no earlier scene showed; "do_not_repeat" lists earlier constructions this scene must not rebuild.
   - The last scene returns to the first scene's picture and answers central_question; "takeaway" is that answer in one sentence.
   - Every scene cites the source_fact_ids its facts come from. Do not invent facts, examples or equations.
   - Accuracy beats neatness. Every learning_goal must be true as stated, for the running example and in general. When a tidy connection holds only in a special case, plan the weaker statement that is always true.
   - Formal proofs: unless the instructions below ask for proofs, keep only the one idea that makes a result believable, shown as a picture inside the scene that needs it. Never plan a scene whose job is to walk through the steps of a proof.
   - Set delivery_mode to manim for every scene.

5. COVERAGE ("coverage"). One entry per source page: "covered" (a scene explains it), "compressed" (a scene states its conclusion without the working), "deferred" (named in passing as beyond this video) or "omitted", with a one-sentence reason for anything not covered. In a short video it is right to leave out side results that do not serve the central question, and to say so here. A full-coverage request in the scene instructions above overrides this: then every substantive page is covered.

CHECK BEFORE YOU ANSWER
- Read the hook aloud as someone who has never seen the document. Is every person, quantity and number introduced before it is used? Does each sentence follow from the one before? Is the snag true for the running example's values (check the arithmetic)?
- Could a viewer say what the central question is after the first twenty seconds?
- Does every scene's visual_objective use the running example or a motif?
- Read the learning_goals in order: is it one argument in which each step needs the one before it?
- Does the last scene answer central_question?
- Is every page in "coverage" once?"""


def hook_check_instructions() -> str:
    """Asks whether each sentence of the planned hook follows; JSON payload follows."""
    return """You check the opening lines of a short explainer video for a viewer who has never seen the topic and hears each sentence once. The hook, the central question it leads to and the running example (with its actual values) are given below.

Read the hook one sentence at a time. A sentence fails when:
- it mentions a person, object, quantity, number or symbol that no earlier sentence (and no part of the same sentence) has introduced;
- it does not follow from the sentence before it: the listener could not say why it comes next (because, so, but);
- it states a conclusion before the setup that makes it believable;
- a claim in it is false for the running example's actual values. Check the arithmetic.

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


SPOKEN_STYLE = f"""SPOKEN NARRATION (it is read aloud by a voice; it must sound like a person talking, not an AI or a textbook):
- Talk the way a good lecturer talks to one student at the board: plain words, contractions (it's, we'll, that's), mostly short sentences with the occasional longer one.
- One idea per sentence. No semicolons, parentheses, em dashes, lists or headings.
- Vary how beats begin; never start two beats in a row with the same word, and do not lean on "So", "Now", "That" or "This" as openers.
- Never use stock phrases: {", ".join(f'"{p}"' for p in STOCK_PHRASES)}. No exclamation marks. At most one question per scene.
- Say math the way you would out loud: "A equals U times D times V transpose", "sigma one", "the length of A v". Never put symbols such as ^T, _k, ||x|| or LaTeX in beat text.
- Names with underscores (motif ids, scene ids) never appear in beat text: say what the thing is ("the gold line", "the cloud of points")."""


def narration_instructions(pace_guidance: str, target_wpm: int, presenter_opens: bool) -> str:
    """What the narration writer is asked to do, before the payload JSON is appended.

    ``pace_guidance`` is planning.py's existing pace sentence(s). The payload
    carries "lesson_spine" (central_question, hook, takeaway, running_example,
    motifs) and each scene's viewer_question.
    """
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
    return f"""Write the requested consecutive section of one spoken lesson. One person speaks it over animation. The lesson plan and its spine (lesson_spine) are settled: the central question, the running example, the motifs and the scene order are not yours to change.

{pace_guidance}

SHAPE
- Each scene opens by making its viewer_question felt, as a question, a snag or a "but", never as a heading, and ends once its learning_goal has been said plainly, one time.
- Show before naming: describe what happens to the running example first; the name or the formula comes afterwards, as a label for what the viewer just saw.
- Whenever a number is said, it is one of the running example's actual values, and a sentence says at most two numbers. Never read out a vector, a list of iterates, a formula with more than three terms or a number with more than three significant figures: round it ("about seventy-eight percent") or say what it does ("the slope settles near one point two").
- A result with a proof in the facts gets the one reason it is believable, in terms of the picture. No lemma-by-lemma steps, no "suppose not", no theorem numbers.
- Opening: {opening}
- Ending: the last scene says the answer to central_question once, points back at the opening picture and stops. No second restatement, no summary list, no sign-off.
- The first beat of every other scene follows straight on from the previous scene's last beat (continuity.previous_scene); never recap it.

FLOW (the listener hears each sentence once and cannot scroll back)
- Each sentence follows from the one before it. When the link is not obvious, say it: "because", "so", "but", "which means".
- Introduce before you use: never mention a person, object, quantity or symbol ("the new student", "beta one", "the gold curve") before the narration, or the picture it describes, has said what it is.
- Set up before you claim: say what the viewer sees first, then what it means. A claim about the running example must be true for its actual values; check the arithmetic before you write it.
- Do not skip a step the viewer needs to follow the argument, and do not take two steps in one sentence.

ECONOMY (this is what separates a script from a transcript of a lecture)
- Say each thing once. Every sentence either tells the viewer something new or tells them where to look.
- No announcements ("now we will see"), no recaps of what was just shown.
- Let the picture carry the detail: do not read out coordinates, matrix entries or every term of a formula that is on screen. Say what to notice about them.
- Prefer the short concrete word: "stretch", not "apply a scaling transformation".
- Accuracy beats neatness: every sentence must be true as stated and agree with the cited facts. When a tidy claim holds only in a special case, say the weaker thing that is always true.

BEATS
- A beat is one visual moment: one to three sentences, 20 to 45 words. When the picture has to change twice, that is two beats.
- visual_action says what CHANGES on screen during the beat, using the motif names: what moves, grows, rotates, morphs, lights up or fades. It must be drawable with dots, arrows, lines, curves, axes, grids, simple shapes and labels of at most four words. Never ask for a sentence, title or caption on screen, and never for photographs or 3D scenes.
- Inside a scene the same motif stays on screen from beat to beat: change it, do not replace it.

{SPOKEN_STYLE}

FORMAT
Return exactly the required_scene_ids once each and in the supplied order. Populate role, pause_after_seconds (normally 0.25-0.55, or 0.6-0.9 for reflection and scene endings), visual_action, continuity_note, transition_in and transition_out. Every beat cites only supplied source_fact_ids. Preserve mathematical notation in the facts you cite, but speak it in words. Set target_words_per_minute to {target_wpm}. Timing fields must be zero; the pipeline computes them after the audio is recorded."""


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
- The narration is heard, so the screen never repeats it: no sentences, titles, headings or captions. Text() is for labels of at most four words beside what they name; mathematics goes in MathTex, at most one new formula per scene, built term by term and colored to match the objects its terms describe.
- Axis titles sit fully inside the frame. A value read off an axis is placed clear of that axis's tick numbers.
- Draw each mark once: an existing bar, dot or line is moved or resized with .animate or a ValueTracker, never drawn again on top of itself.
- Pace: Transform and moves take at least 1.0 s, Create and Write at least 0.6 s. Two or three unhurried changes per beat teach more than six quick ones.
- Keep the picture alive while the narrator talks: never hold the frame unchanged for more than about five seconds. When a beat runs long, spread its changes across it, or use the extra time for a small motion that points at what is being said (a dot travelling along the curve, a highlight sweeping the term, a ValueTracker easing a value).
- When "visual_world" is given, draw its running example with exactly those values and give each motif its stated color and meaning, so every scene of the video shares one look."""


# ── narration review ────────────────────────────────────────────────────

REVIEW_PASS_SCORE = 6.5
# A script also fails review when its logical_flow score is below this, or
# when the reviewer lists a sentence that does not follow from the last.
FLOW_PASS_SCORE = 6.0
