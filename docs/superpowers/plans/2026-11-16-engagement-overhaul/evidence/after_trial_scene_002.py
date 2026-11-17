"""
Scene: scene_002
Narration: Keep sweeping and watch the tally bar. It peaks exactly when the line cuts through the long diagonal...
Duration: 30.7s
Generated: 2026-11-16T20:14:56.405105
"""

from manim import *
from src.utils.coordinate_system_helpers import attach_axis_labels, create_coordinate_system, plot_function_on_system
from src.utils.notation_helpers import create_notation, create_notation_transform_target, create_text_label
from src.utils.pi_characters import create_character_for_role, create_character_lineup, create_pi_teacher, create_pi_students, create_speech_bubble
from src.utils.animation_patterns import progressive_write, dim_and_focus, staged_transform, execute_animation_steps
from src.utils.layout_tracker import LayoutTracker
from src.utils.geometry_helpers import connect, drop_perpendicular, on_axes, perpendicular_foot
from src.utils.narration_sync import sync_to, sync_to_end

class Scene_002(Scene):
    def construct(self):
        layout = LayoutTracker()

        center = np.array([-1.2, 0.0, 0.0])
        optimal_angle = np.arctan2(2.1, 3.0)
        u_dir = np.array([np.cos(optimal_angle), np.sin(optimal_angle), 0.0])
        v_dir = np.array([-np.sin(optimal_angle), np.cos(optimal_angle), 0.0])

        # Purpose [beat_004, beat_005, beat_006]: The data points whose diagonal alignment defines the winning direction.
        pt1 = center - 2.1 * u_dir - 0.25 * v_dir
        pt2 = center - 0.7 * u_dir + 0.35 * v_dir
        pt3 = center + 0.6 * u_dir - 0.40 * v_dir
        pt4 = center + 2.0 * u_dir + 0.30 * v_dir
        point_cloud = VGroup(*(Dot(p, radius=0.1, color=WHITE) for p in [pt1, pt2, pt3, pt4]))
        layout.register("point_cloud", point_cloud)

        # Purpose [beat_004, beat_005]: The score bar that rises as the candidate line improves and peaks at the best fit.
        tally_bg = RoundedRectangle(
            corner_radius=0.1,
            width=0.7,
            height=3.4,
            stroke_color=GRAY_C,
            stroke_width=2,
            fill_color=BLACK,
            fill_opacity=0.6,
        ).move_to(RIGHT * 4.8 + UP * 0.2)
        layout.register("tally_bg", tally_bg)

        tally_track_base = tally_bg.get_bottom() + UP * 0.15
        max_bar_height = 3.0

        theta_tracker = ValueTracker(optimal_angle - 0.75)

        def get_tally_height():
            diff = theta_tracker.get_value() - optimal_angle
            factor = max(0.18, np.cos(diff) ** 4)
            return factor * max_bar_height

        projection_tally = always_redraw(
            lambda: Rectangle(
                width=0.48,
                height=get_tally_height(),
                fill_color=BLUE_B if abs(theta_tracker.get_value() - optimal_angle) > 0.08 else GOLD_C,
                fill_opacity=0.85,
                stroke_width=0,
            ).move_to(tally_track_base + UP * (get_tally_height() / 2.0))
        )
        layout.register("projection_tally", projection_tally)

        # Purpose [beat_004]: The sweeping tested direction that rotates until it best matches the data.
        line_half_len = 3.6
        candidate_line = always_redraw(
            lambda: Line(
                start=center - line_half_len * np.array([np.cos(theta_tracker.get_value()), np.sin(theta_tracker.get_value()), 0.0]),
                end=center + line_half_len * np.array([np.cos(theta_tracker.get_value()), np.sin(theta_tracker.get_value()), 0.0]),
                color=GRAY_B,
                stroke_width=4,
            )
        )
        layout.register("candidate_line", candidate_line)

        # Initial appearance
        self.play(
            FadeIn(point_cloud, shift=UP * 0.2),
            Create(tally_bg),
            Create(candidate_line),
            FadeIn(projection_tally),
            run_time=1.0,
        )
        self.wait(0.9)

        # Sweep candidate line toward optimal diagonal alignment while tally bar climbs
        self.play(
            theta_tracker.animate.set_value(optimal_angle),
            run_time=6.5,
            rate_func=smooth,
        )
        self.wait(0.9)

        # Lock in candidate line: freeze dynamic redraws into fixed objects
        candidate_line.clear_updaters()
        projection_tally.clear_updaters()

        # Purpose [beat_004, beat_005, beat_006]: The winning direction through the points, revealing the top singular vector.
        best_line_v1 = Arrow(
            start=center - 3.2 * u_dir,
            end=center + 3.4 * u_dir,
            color=GOLD_C,
            stroke_width=6,
            buff=0,
            max_tip_length_to_length_ratio=0.15,
        )
        layout.register("best_line_v1", best_line_v1)
        layout.unregister("candidate_line")

        static_tally = Rectangle(
            width=0.48,
            height=max_bar_height,
            fill_color=GOLD_C,
            fill_opacity=0.9,
            stroke_width=0,
        ).move_to(tally_track_base + UP * (max_bar_height / 2.0))

        self.play(
            ReplacementTransform(candidate_line, best_line_v1),
            ReplacementTransform(projection_tally, static_tally),
            run_time=1.5,
        )
        self.play(
            Indicate(best_line_v1, color=GOLD_A, scale_factor=1.03),
            run_time=1.2,
        )
        self.wait(0.9)

        # Purpose [beat_005]: Marks the peak of the tally to connect the best fit with the highest score.
        peak_highlight = Circle(
            radius=0.35,
            color=YELLOW_C,
            stroke_width=3,
        ).move_to(static_tally.get_top())
        layout.register("peak_highlight", peak_highlight)

        # Purpose [beat_005]: Names the winning direction without adding extra explanation.
        top_singular_vector_label = Text("top singular vector", font_size=28, color=GOLD_C)
        layout.place_label(top_singular_vector_label, "top_singular_vector_label", anchor="best_line_v1", preferred=UP, buff=0.35)

        self.play(
            Create(peak_highlight),
            Write(top_singular_vector_label),
            run_time=1.2,
        )

        # Purpose [beat_005]: Shows the name of the first singular value tied to the peak score.
        singular_value_formula = MathTex(r"\sigma_1", color=GOLD_C, font_size=40)
        layout.place_label(singular_value_formula, "singular_value_formula", anchor="peak_highlight", preferred=LEFT, buff=0.3)

        self.play(
            Write(singular_value_formula),
            Circumscribe(peak_highlight, color=GOLD_B, run_time=1.4),
        )
        self.wait(1.6)

        # Clear side tally and formulas to open space for residual analysis
        layout.unregister("peak_highlight")
        layout.unregister("top_singular_vector_label")
        layout.unregister("singular_value_formula")
        layout.unregister("tally_bg")
        layout.unregister("projection_tally")

        self.play(
            FadeOut(tally_bg),
            FadeOut(static_tally),
            FadeOut(peak_highlight),
            FadeOut(top_singular_vector_label),
            FadeOut(singular_value_formula),
            run_time=1.2,
        )

        # Purpose [beat_006]: A faint band of leftover spread off the main line.
        poly_pts = [
            pt1 - 0.3 * v_dir,
            pt2 + 0.4 * v_dir,
            pt4 + 0.35 * v_dir,
            pt4 - 0.25 * v_dir,
            pt3 - 0.4 * v_dir,
            pt1 + 0.25 * v_dir,
        ]
        residual_spread = Polygon(
            *poly_pts,
            stroke_color=GRAY_B,
            stroke_width=1.5,
            stroke_opacity=0.6,
            fill_color=GRAY_B,
            fill_opacity=0.15,
        )
        layout.register("residual_spread", residual_spread)

        residual_drops = VGroup()
        for pt in [pt1, pt2, pt3, pt4]:
            proj_point = center + np.dot(pt - center, u_dir) * u_dir
            residual_drops.add(
                DashedLine(pt, proj_point, color=GRAY_B, stroke_width=2, dash_length=0.08)
            )

        self.play(
            FadeIn(residual_spread),
            Create(residual_drops),
            run_time=1.5,
        )
        self.wait(1.1)

        # Purpose [beat_006]: Shows the next best direction must be perpendicular to avoid measuring the same spread twice.
        perpendicular_guide = DashedLine(
            start=center - 2.5 * v_dir,
            end=center + 2.5 * v_dir,
            color=PURPLE_C,
            stroke_width=3,
            dash_length=0.12,
        )
        layout.register("perpendicular_guide", perpendicular_guide)

        elbow_size = 0.3
        right_angle = VGroup(
            Line(center + elbow_size * u_dir, center + elbow_size * u_dir + elbow_size * v_dir, color=PURPLE_B, stroke_width=2),
            Line(center + elbow_size * u_dir + elbow_size * v_dir, center + elbow_size * v_dir, color=PURPLE_B, stroke_width=2),
        )

        self.play(
            Create(perpendicular_guide),
            Create(right_angle),
            run_time=1.5,
        )
        self.wait(0.9)

        # Purpose [beat_006]: The perpendicular second singular vector that captures the remaining spread.
        second_axis_v2 = Arrow(
            start=center,
            end=center + 2.1 * v_dir,
            color=PURPLE_C,
            stroke_width=6,
            buff=0,
            max_tip_length_to_length_ratio=0.18,
        )
        layout.register("second_axis_v2", second_axis_v2)

        v2_label = MathTex(r"\mathbf{v}_2", color=PURPLE_C, font_size=34)
        layout.place_label(v2_label, "v2_label", anchor="second_axis_v2", preferred=UL, buff=0.25)

        self.play(
            GrowArrow(second_axis_v2),
            run_time=1.5,
        )
        self.play(
            Write(v2_label),
            run_time=1.0,
        )
        self.wait(1.6)

        # Clean non-carry-out objects to leave point_cloud, best_line_v1, second_axis_v2
        layout.unregister("residual_spread")
        layout.unregister("perpendicular_guide")
        layout.unregister("v2_label")

        self.play(
            FadeOut(residual_spread),
            FadeOut(residual_drops),
            FadeOut(perpendicular_guide),
            FadeOut(right_angle),
            FadeOut(v2_label),
            run_time=1.2,
        )
        self.wait(1.9)

        # Total animation time: 30.5s