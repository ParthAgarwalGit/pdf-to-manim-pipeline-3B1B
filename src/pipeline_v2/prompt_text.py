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


def lesson_instructions(scene_guidance: str, presenter_opens: bool) -> str:
    """What the lesson planner is asked to do, before the facts JSON is appended.

    ``scene_guidance`` is planning.py's existing scene-count sentence(s)
    (full-coverage or short-document wording). ``presenter_opens``: an
    on-camera presenter speaks the hook before the first animation.
    """
    if presenter_opens:
        hook_rule = (
            'A presenter speaks "hook" on camera before any animation, so it must work with no '
            "picture: first person, at most 35 words, ending on central_question without "
            "answering it. The hook is not a scene. The first scene opens directly on the running "
            "example and makes the question concrete."
        )
    else:
        hook_rule = (
            '"hook" is the first thing the narrator says, over the first scene\'s opening '
            "picture: at most 40 words, no greeting, no topic announcement, ending on "
            "central_question without answering it."
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
- Could a viewer say what the central question is after the first twenty seconds?
- Does every scene's visual_objective use the running example or a motif?
- Read the learning_goals in order: is it one argument in which each step needs the one before it?
- Does the last scene answer central_question?
- Is every page in "coverage" once?"""


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
- Whenever a number is said, it is one of the running example's actual values.
- A result with a proof in the facts gets the one reason it is believable, in terms of the picture. No lemma-by-lemma steps, no "suppose not", no theorem numbers.
- Opening: {opening}
- Ending: the last scene says the answer to central_question once, points back at the opening picture and stops. No second restatement, no summary list, no sign-off.
- The first beat of every other scene follows straight on from the previous scene's last beat (continuity.previous_scene); never recap it.

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


def tts_direction(audience: Optional[str] = None) -> str:
    """Delivery direction prepended to every Gemini TTS request (not spoken)."""
    listener = f"a {audience} student" if audience else "one student"
    return (
        f"Read the following as a warm, curious teacher explaining an idea she finds genuinely "
        f"interesting to {listener}, sitting beside them. Natural rise and fall in pitch, a small "
        "lift on the surprising or important word in each sentence, a brief pause at each comma "
        "and a longer one between paragraphs. Unhurried but never slow, never flat, never "
        "announcer-like. Do not add, skip or change any words."
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
- When "visual_world" is given, draw its running example with exactly those values and give each motif its stated color and meaning, so every scene of the video shares one look."""


# ── narration review ────────────────────────────────────────────────────

REVIEW_PASS_SCORE = 6.5
