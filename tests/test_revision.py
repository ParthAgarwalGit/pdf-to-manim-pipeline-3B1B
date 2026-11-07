"""P1 of the professor-control design: typed beats, lesson_script.json,
per-scene fingerprints and incremental revision runs.

The media stages are faked (no manim, ffmpeg or TTS API): the fakes write
small files and record which scenes they were asked to render or narrate,
which is what the reuse assertions check.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.schemas.animation_code import AnimationCodeObject

from src.agents.scene_controller import SceneControllerAgent, SceneControllerInput
from src.agents.script_architect import (
    ScriptArchitectAgent,
    _parse_analogies,
    _typed_beat_fields,
)
from src.agents.tts_agent import TTSOutput
from src.agents.video_renderer import SceneRenderResult, VideoRendererOutput
from src.orchestrator.pipeline import Orchestrator, PipelineConfig
from src.orchestrator.revision import (
    code_object_from_file,
    load_base_run,
    next_revision_dir,
)
from src.schemas.source_segment import SourceSegment
from src.schemas.storyboard import Configuration, PDFMetadataStoryboard, StoryboardJSON
from src.schemas.video_script import Analogy, Beat, VideoScript
from src.utils.fingerprints import beat_fingerprint


# ── fixtures ────────────────────────────────────────────────────────────

def _config() -> Configuration:
    return Configuration(verbosity="medium", depth="intermediate", audience="undergraduate")


def _beat(n: int, segment: str, text: str = "", **extra) -> Beat:
    return Beat(
        beat_id=f"beat_{n:03d}",
        narration=text or f"Beat {n} explains idea {n}. " + "More detail here. " * 10,
        visual_hint=f"diagram for idea {n}",
        introduces_concepts=[f"concept_{n}"],
        source_segment_id=segment,
        **extra,
    )


def _script() -> VideoScript:
    return VideoScript(
        title="Notes",
        beats=[
            _beat(1, "seg_a"), _beat(2, "seg_a"),
            _beat(3, "seg_b"), _beat(4, "seg_b"), _beat(5, "seg_b"), _beat(6, "seg_b"),
            _beat(7, "seg_c"), _beat(8, "seg_c"),
        ],
    )


def _segment_scene_ids(storyboard, segment: str):
    return {
        s.scene_id for s in storyboard.scenes
        if s.source_beats and s.source_beats[0].source_segment_id == segment
    }


class FakeMedia:
    """Stands in for code generation, manim, ffmpeg and the TTS API."""

    def __init__(self, orchestrator: Orchestrator):
        self.generated = []
        self.rendered = []
        self.narrated = []
        orchestrator.scene_controller_agent._process_scene_with_generator = self.codegen
        renderer = orchestrator.video_renderer_agent
        renderer.execute = self.render
        renderer._concatenate_videos = self.concat
        orchestrator.tts_agent.process_all_scenes = self.tts
        orchestrator._run_audio_video_composition = lambda *a, **k: None

    def codegen(self, generator, processing_input, scene_id, review_feedback=None):
        self.generated.append(scene_id)
        spec = processing_input.scene_spec
        return AnimationCodeObject(
            scene_id=scene_id,
            code=(
                f"class Scene{scene_id.split('_')[-1]}(Scene):\n"
                "    def construct(self):\n"
                "        self.add(Circle())\n"
                "        self.wait(1)"
            ),
            imports=["from manim import *"],
            class_name=f"Scene{scene_id.split('_')[-1]}",
            narration=spec.narration,
            duration_estimate=max(float(spec.duration_estimate), 1.0),
            generation_timestamp=datetime.now(timezone.utc),
        )

    def render(self, input_data):
        results = []
        for scene_file in input_data.scene_files:
            stem = Path(scene_file).stem
            scene_id = stem[len("scene_"):]
            video = Path(input_data.output_dir) / "media" / "videos" / stem / "720p30" / "S.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"mp4:" + scene_id.encode())
            self.rendered.append(scene_id)
            results.append(SceneRenderResult(
                scene_id=scene_id, success=True, video_path=str(video), duration_seconds=10.0,
            ))
        final = Path(input_data.output_dir) / input_data.final_video_name
        final.write_bytes(b"concat")
        return VideoRendererOutput(
            final_video_path=str(final), scene_results=results, total_duration=10.0 * len(results),
        )

    def concat(self, paths, output):
        Path(output).write_bytes(b"concat:" + str(len(paths)).encode())

    def tts(self, scenes, output_dir, quality="standard", voice="alloy"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = []
        for scene in scenes:
            # Same layout as TTSAgent.process_all_scenes: audio/<id>/<id>.mp3
            audio = Path(output_dir) / scene["scene_id"] / f"{scene['scene_id']}.mp3"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(f"{voice}:{scene['narration']}".encode())
            self.narrated.append(scene["scene_id"])
            beats = scene.get("beats") or []
            # Voice-dependent pacing, so a new voice means a new timing track.
            per_beat = 9.0 / max(1, len(beats)) * (1.1 if voice == "nova" else 1.0)
            timings = {b: [round(i * per_beat, 3), round((i + 1) * per_beat, 3)] for i, (b, _) in enumerate(beats)}
            results.append(TTSOutput(
                audio_file_path=str(audio), duration_seconds=round(per_beat * max(1, len(beats)), 3),
                scene_id=scene["scene_id"], beat_timings=timings,
            ))
        return results


def _orchestrator(tmp_path, **config) -> Orchestrator:
    orchestrator = Orchestrator(config=PipelineConfig(
        base_output_dir=str(tmp_path),
        run_id="run1",
        enable_final_video_review=False,
        **config,
    ))
    orchestrator.media = FakeMedia(orchestrator)
    return orchestrator


def _base_run(tmp_path, script: VideoScript, **config):
    """A real run_pipeline from the storyboard stage on (front half stubbed)."""
    orchestrator = _orchestrator(tmp_path, **config)
    ref = SimpleNamespace(filename="notes.pdf")
    orchestrator._run_ingestion = lambda *a, **k: ref
    orchestrator._run_validation = lambda *a, **k: ref
    orchestrator._run_vision_ocr = lambda *a, **k: None
    orchestrator._run_document_segmenter = lambda *a, **k: []

    def fake_architect(ocr_output, structured_logger, source_segments=None):
        agent = orchestrator.script_architect_agent
        planned = script.model_copy(deep=True)
        scenes = agent._chunk_script_into_scenes(planned)
        agent._plan_visual_intents(scenes, planned, _config(), [])
        agent._renumber_scenes(scenes)
        agent.last_video_script = planned
        return StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="notes.pdf", page_count=1),
            configuration=_config(),
            concept_hierarchy=agent._concept_hierarchy_from_ledger(planned.concept_ledger()),
            scenes=scenes,
        )

    orchestrator._run_script_architect = fake_architect
    manifest = orchestrator.run_pipeline("notes.pdf", event_source="local")
    return Path(manifest.output_directory), manifest


def _digest_tree(root: Path):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "rev_" not in p.relative_to(root).as_posix()
    }


def _edited(script: VideoScript, beat_id: str, narration: str) -> VideoScript:
    edited = script.model_copy(deep=True)
    for beat in edited.beats:
        if beat.beat_id == beat_id:
            beat.narration = narration
            beat.author, beat.locked = "professor", True
    return edited


# ── typed beats ─────────────────────────────────────────────────────────

class TestTypedBeats:
    def test_old_beat_json_loads_with_defaults(self):
        beat = Beat.model_validate({"beat_id": "beat_001", "narration": "x", "visual_hint": "y"})
        assert (beat.role, beat.depth, beat.analogy_id, beat.author, beat.locked) == (
            "core", 0, None, "generated", False,
        )

    def test_undeclared_analogy_gets_stub(self):
        script = VideoScript(beats=[_beat(1, "", analogy_id="analogy_water_pipes", role="analogy")])
        assert [a.analogy_id for a in script.analogies] == ["analogy_water_pipes"]
        assert script.analogies[0].label == "water pipes"

    def test_duplicate_analogy_id_rejected(self):
        with pytest.raises(ValueError, match="duplicate analogy_id"):
            VideoScript(
                beats=[_beat(1, "")],
                analogies=[Analogy(analogy_id="analogy_a"), Analogy(analogy_id="analogy_a")],
            )

    def test_llm_tags_are_coerced(self):
        fields = _typed_beat_fields(
            {"role": "Bridge", "depth": "9", "analogy_id": "Water Pipes"}, {},
        )
        # bridge belongs to the presenter layer, not the script LLM
        assert fields == {"role": "core", "depth": 3, "analogy_id": "analogy_water_pipes"}
        assert _typed_beat_fields({"depth": "deep"}, {})["depth"] == 0

    def test_analogy_callback_and_collision(self):
        known = {}
        first = _parse_analogies([{"analogy_id": "pipes", "label": "water pipes"}], known)
        again = _parse_analogies(
            [
                {"analogy_id": "pipes", "label": "Water Pipes"},   # callback
                {"analogy_id": "pipes", "label": "drain pipes"},   # different analogy
            ],
            known,
        )
        assert first == {"pipes": "analogy_pipes"}
        assert list(known) == ["analogy_pipes", "analogy_pipes_2"]
        assert again["pipes"] == "analogy_pipes_2"

    def test_segmented_script_tags_beats_and_shares_analogies(self):
        agent = ScriptArchitectAgent()
        responses = [
            {"beats": [
                {"beat_id": "b1", "narration": "Voltage is like water pressure.", "visual_hint": "pipes",
                 "role": "analogy", "depth": 0, "analogy_id": "analogy_pipes"},
                {"beat_id": "b2", "narration": "Formally, V equals IR.", "visual_hint": "equation",
                 "role": "derivation", "depth": 1, "analogy_id": None},
             ],
             "analogies": [{"analogy_id": "analogy_pipes", "label": "water in pipes",
                            "target_concepts": ["Voltage"], "mapping": {"pressure": "voltage"}}]},
            {"beats": [
                {"beat_id": "b1", "narration": "Back to our pipes: current is flow.", "visual_hint": "flow",
                 "role": "analogy", "depth": 0, "analogy_id": "analogy_pipes"},
             ],
             "analogies": []},
        ]
        prompts = []

        def create(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            body = json.dumps(responses[len(prompts) - 1])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])

        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        segments = [
            SourceSegment(segment_id="seg_001", order=0, page_start=1, page_end=1, text="voltage"),
            SourceSegment(segment_id="seg_002", order=1, page_start=2, page_end=2, text="current"),
        ]
        script = agent._generate_segmented_video_script(segments, _config(), [])

        assert [(b.role, b.depth, b.analogy_id) for b in script.beats] == [
            ("analogy", 0, "analogy_pipes"), ("derivation", 1, None), ("analogy", 0, "analogy_pipes"),
        ]
        assert [a.analogy_id for a in script.analogies] == ["analogy_pipes"]
        assert script.analogies[0].target_concepts == ["voltage"]
        assert "ANALOGIES ALREADY INTRODUCED" not in prompts[0]
        assert "analogy_pipes: water in pipes" in prompts[1]


# ── fingerprints and helpers ────────────────────────────────────────────

class TestFingerprintsAndHelpers:
    def test_retagging_a_beat_keeps_its_fingerprint(self):
        beat = _beat(1, "seg_a")
        retagged = beat.model_copy(update={
            "role": "example", "depth": 2, "author": "professor", "locked": True,
        })
        reworded = beat.model_copy(update={"narration": "Different words."})
        assert beat_fingerprint(retagged) == beat_fingerprint(beat)
        assert beat_fingerprint(reworded) != beat_fingerprint(beat)

    def test_code_object_from_file_drops_header(self, tmp_path):
        path = tmp_path / "scene_scene_004.py"
        path.write_text(
            '"""\nScene: scene_004\nNarration:\nold words\n"""\n\n'
            "from manim import *\n\nclass Helper:\n    pass\n\n"
            "class Scene004(Scene):\n    def construct(self):\n        pass\n",
            encoding="utf-8",
        )
        scene = SimpleNamespace(scene_id="scene_002", narration="new words", duration_estimate=12.0)
        code = code_object_from_file(path, scene)
        assert code.class_name == "Scene004"
        assert code.imports == ["from manim import *"]
        assert code.code.startswith("class Helper:")
        assert "old words" not in code.code
        assert code.scene_id == "scene_002"

    def test_revision_dirs_number_from_the_original_run(self, tmp_path):
        run = tmp_path / "run1"
        run.mkdir()
        assert next_revision_dir(run).name == "rev_002"
        rev = run / "rev_002"
        rev.mkdir()
        (rev / "revision.json").write_text("{}", encoding="utf-8")
        assert next_revision_dir(rev) == run / "rev_003"

    def test_scene_controller_generates_only_selected_scenes(self):
        agent = ScriptArchitectAgent()
        script = _script()
        scenes = agent._chunk_script_into_scenes(script)
        agent._plan_visual_intents(scenes, script, _config(), [])
        agent._renumber_scenes(scenes)
        storyboard = StoryboardJSON(
            pdf_metadata=PDFMetadataStoryboard(filename="n.pdf", page_count=1),
            configuration=_config(),
            concept_hierarchy=agent._concept_hierarchy_from_ledger(script.concept_ledger()),
            scenes=scenes,
        )
        target = scenes[-1].scene_id
        result = SceneControllerAgent().execute(SceneControllerInput(
            storyboard=storyboard, output_directory="out", only_scene_ids=[target],
        ))
        assert [c.scene_id for c in result.animation_code_objects] == [target]
        assert result.report.total_scenes == 1
        # Skipped scenes still taught their concepts to the shared context.
        assert "concept_1" in result.shared_context.narrative.concepts_introduced


# ── end-to-end: base run, then revisions ────────────────────────────────

class TestRevisionRuns:
    def test_base_run_persists_lesson_script_and_scene_records(self, tmp_path):
        run_dir, manifest = _base_run(tmp_path, _script())

        saved = VideoScript.model_validate_json((run_dir / "lesson_script.json").read_text("utf-8"))
        assert saved.title == "Notes"
        assert [b.beat_id for b in saved.beats] == [b.beat_id for b in _script().beats]
        assert "lesson_script" in {f.type for f in manifest.files}
        assert manifest.run_config.tts_first is False
        assert len(manifest.scenes) == manifest.total_scenes
        for record in manifest.scenes:
            assert (run_dir / record.code_path).is_file()
            assert (run_dir / record.video_path).is_file()
            assert (run_dir / record.audio_path).is_file()
            assert record.reused_video_from is None

    def test_editing_one_beat_rebuilds_only_its_scene(self, tmp_path):
        script = _script()
        run_dir, _ = _base_run(tmp_path, script)
        base = load_base_run(run_dir)
        edited_scene = next(
            s.scene_id for s in base.storyboard.scenes
            if "beat_006" in [b.beat_id for b in s.source_beats]
        )
        before = _digest_tree(run_dir)

        orchestrator = _orchestrator(tmp_path)
        manifest = orchestrator.run_revision(
            str(run_dir), _edited(script, "beat_006", "A professor's own words about idea six."),
        )

        rev_dir = Path(manifest.output_directory)
        assert rev_dir == run_dir / "rev_002"
        assert _digest_tree(run_dir) == before, "the base run must never be modified"
        assert orchestrator.media.generated == [edited_scene]
        assert orchestrator.media.rendered == [edited_scene]
        assert orchestrator.media.narrated == [edited_scene]

        report = json.loads((rev_dir / "revision.json").read_text("utf-8"))
        assert report["beats"] == {"added": [], "removed": [], "changed": ["beat_006"]}
        assert report["scenes"]["regenerated"] == [edited_scene]
        assert len(report["scenes"]["reused_video"]) == manifest.total_scenes - 1

        # The revision is a complete run on its own.
        assert manifest.successful_scenes == manifest.total_scenes
        assert (rev_dir / "final_output.mp4").is_file()
        for record in manifest.scenes:
            assert (rev_dir / record.code_path).is_file()
            assert (rev_dir / record.video_path).is_file()
            assert (rev_dir / record.audio_path).is_file()
        saved = VideoScript.model_validate_json((rev_dir / "lesson_script.json").read_text("utf-8"))
        edited_beat = next(b for b in saved.beats if b.beat_id == "beat_006")
        assert edited_beat.locked and edited_beat.author == "professor"

    def test_inserted_beat_shifts_ids_but_reuses_other_segments(self, tmp_path):
        script = _script()
        run_dir, _ = _base_run(tmp_path, script)
        base = load_base_run(run_dir)
        edited = script.model_copy(deep=True)
        edited.beats.insert(2, _beat(99, "seg_a", text="An extra example for segment A."))

        orchestrator = _orchestrator(tmp_path)
        manifest = orchestrator.run_revision(str(run_dir), edited)

        report = json.loads((Path(manifest.output_directory) / "revision.json").read_text("utf-8"))
        assert report["beats"]["added"] == ["beat_099"]
        reused_from = set(report["scenes"]["reused_video"].values())
        # Segments B and C are untouched, so every one of their scenes is reused
        # even though their scene ids shift.
        assert _segment_scene_ids(base.storyboard, "seg_b") | _segment_scene_ids(
            base.storyboard, "seg_c"
        ) <= reused_from
        assert not _segment_scene_ids(base.storyboard, "seg_a") & reused_from

    def test_voice_change_renarrates_but_keeps_renders(self, tmp_path):
        script = _script()
        run_dir, base_manifest = _base_run(tmp_path, script)

        orchestrator = _orchestrator(tmp_path, tts_voice="nova")
        manifest = orchestrator.run_revision(str(run_dir), script)

        assert orchestrator.media.generated == [] and orchestrator.media.rendered == []
        assert len(orchestrator.media.narrated) == base_manifest.total_scenes
        assert all(r.reused_video_from and not r.reused_audio_from for r in manifest.scenes)

    def test_dry_run_plans_without_spending(self, tmp_path):
        script = _script()
        run_dir, _ = _base_run(tmp_path, script)

        orchestrator = _orchestrator(tmp_path, dry_run=True)
        manifest = orchestrator.run_revision(str(run_dir), _edited(script, "beat_001", "New opening."))

        assert orchestrator.media.rendered == [] and orchestrator.media.narrated == []
        report = json.loads((Path(manifest.output_directory) / "revision.json").read_text("utf-8"))
        assert report["dry_run"] is True
        assert len(report["scenes"]["regenerated"]) == 1

    def test_revision_of_a_revision_lands_beside_it(self, tmp_path):
        script = _script()
        run_dir, _ = _base_run(tmp_path, script)
        first = _orchestrator(tmp_path).run_revision(
            str(run_dir), _edited(script, "beat_008", "Closing words."),
        )

        orchestrator = _orchestrator(tmp_path)
        second = orchestrator.run_revision(first.output_directory, _edited(script, "beat_008", "Closing words."))

        assert Path(second.output_directory) == run_dir / "rev_003"
        assert orchestrator.media.rendered == [] and orchestrator.media.narrated == []

    def test_tts_first_new_narration_forces_new_render(self, tmp_path):
        script = _script()
        run_dir, base_manifest = _base_run(tmp_path, script, tts_first=True)
        assert base_manifest.run_config.tts_first is True

        unchanged = _orchestrator(tmp_path, tts_first=True)
        unchanged.run_revision(str(run_dir), script)
        assert unchanged.media.rendered == [] and unchanged.media.narrated == []

        revoiced = _orchestrator(tmp_path, tts_first=True, tts_voice="nova")
        manifest = revoiced.run_revision(str(run_dir), script)
        # A new recording re-times every scene, so every render is redone,
        # but the code is reused with the new timing track injected: no codegen.
        assert len(revoiced.media.rendered) == base_manifest.total_scenes
        assert len(revoiced.media.narrated) == base_manifest.total_scenes
        assert revoiced.media.generated == []
        rev_dir = Path(manifest.output_directory)
        report = json.loads((rev_dir / "revision.json").read_text("utf-8"))
        assert len(report["scenes"]["retimed"]) == base_manifest.total_scenes
        first = manifest.scenes[0]
        code = (rev_dir / first.code_path).read_text("utf-8")
        assert "BEAT_TIMES = {" in code and code.rstrip().endswith("sync_to_end(self)")

    def test_legacy_run_without_records_is_still_revisable(self, tmp_path):
        script = _script()
        run_dir, base_manifest = _base_run(tmp_path, script)
        manifest_path = run_dir / "manifest.json"
        legacy = json.loads(manifest_path.read_text("utf-8"))
        legacy.pop("scenes")
        manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
        (run_dir / "lesson_script.json").unlink()

        base = load_base_run(run_dir)
        assert set(base.records) == {s.scene_id for s in base.storyboard.scenes}

        orchestrator = _orchestrator(tmp_path)
        orchestrator.run_revision(str(run_dir), script)
        # Code and audio are found at their conventional paths; video too.
        assert orchestrator.media.rendered == [] and orchestrator.media.narrated == []


class TestReviseCommand:
    def test_revise_keeps_the_parent_runs_voice_and_quality(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        import src.cli as cli

        run_dir, _ = _base_run(tmp_path, _script(), tts_voice="nova", render_quality="high")
        edited = _edited(_script(), "beat_002", "Edited words.")
        script_file = tmp_path / "edits.json"
        script_file.write_text(edited.model_dump_json(), encoding="utf-8")
        captured = {}

        def fake_run_revision(self, base_run_dir, script, **kwargs):
            captured["config"], captured["script"] = self.config, script
            rev = Path(base_run_dir) / "rev_002"
            rev.mkdir()
            (rev / "revision.json").write_text(json.dumps({
                "beats": {"added": [], "removed": [], "changed": ["beat_002"]},
                "scenes": {"total": 4, "regenerated": ["scene_001"],
                           "reused_video": {}, "reused_audio": {}},
                "cost_usd": 0.0,
            }), encoding="utf-8")
            return SimpleNamespace(output_directory=str(rev), status="SUCCESS", final_video_path=None)

        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        monkeypatch.setattr(Orchestrator, "run_revision", fake_run_revision)
        result = CliRunner().invoke(
            cli.pdf2manim, ["revise", str(run_dir), "--script", str(script_file)],
        )

        assert result.exit_code == 0, result.output
        assert captured["config"].tts_voice == "nova"
        assert captured["config"].render_quality == "high"
        assert captured["script"].beats[1].narration == "Edited words."
        assert "1 changed" in result.output

    def test_revise_rejects_a_non_run_directory(self, tmp_path, monkeypatch):
        from click.testing import CliRunner

        import src.cli as cli

        monkeypatch.setattr(cli, "_load_env_files", lambda: None)
        result = CliRunner().invoke(cli.pdf2manim, ["revise", str(tmp_path)])
        assert result.exit_code != 0
        assert "storyboard.json" in result.output
