# Instrumented automatically by PDF-to-Manim V2.
from manim import *
import numpy as np
from src.pipeline_v2.tracked_scene import TrackedScene

class V2Scene4WheelExample(TrackedScene):

    def construct(self):
        self.configure_generated_scene(layout_template='full_visual', target_duration=21.496292, beat_schedule=[('scene_4_wheel_example_beat_1', 0.0, 8.679), ('scene_4_wheel_example_beat_2', 8.679, 16.069), ('scene_4_wheel_example_beat_3', 16.069, 21.496)])
        ground_y = -1.6
        self.auto_register('ground_y', ground_y)
        wheel_radius = 0.9
        self.auto_register('wheel_radius', wheel_radius)
        start_x = -4.5
        self.auto_register('start_x', start_x)
        one_rev_dist = 2 * np.pi * wheel_radius
        self.auto_register('one_rev_dist', one_rev_dist)
        ground = Line(start=np.array([-7.0, ground_y, 0]), end=np.array([7.0, ground_y, 0]), color=GREY_B, stroke_width=3)
        self.auto_register('ground', ground)
        road_surface = Rectangle(width=14.0, height=1.2, fill_color=DARK_GRAY, fill_opacity=0.3, stroke_width=0).next_to(ground, DOWN, buff=0)
        self.auto_register('road_surface', road_surface)
        ground_ticks = VGroup(*[Line(np.array([x, ground_y, 0]), np.array([x - 0.2, ground_y - 0.25, 0]), color=GREY_C, stroke_width=1.5) for x in np.arange(-6.8, 6.9, 0.4)])
        self.auto_register('ground_ticks', ground_ticks)
        header = Text('Rolling Wheel: One Turn = One Circumference', font_size=30, color=WHITE)
        self.auto_register('header', header)
        header.to_edge(UP, buff=0.45)

        def create_wheel():
            tire = Circle(radius=wheel_radius, color=BLUE_D, stroke_width=7)
            inner_rim = Circle(radius=wheel_radius * 0.88, color=TEAL_A, stroke_width=2)
            hub = Dot(ORIGIN, radius=0.08, color=WHITE)
            spokes = VGroup()
            for i in range(8):
                angle = i * (TAU / 8)
                spoke = Line(ORIGIN, np.array([np.cos(angle), np.sin(angle), 0]) * (wheel_radius * 0.88), color=GREY_A, stroke_width=1.8)
                spokes.add(spoke)
            contact_marker = Dot(np.array([0, -wheel_radius, 0]), radius=0.09, color=YELLOW)
            wheel_vg = VGroup(tire, inner_rim, spokes, hub, contact_marker)
            return wheel_vg
        wheel = create_wheel()
        self.auto_register('wheel', wheel)
        wheel.move_to(np.array([start_x, ground_y + wheel_radius, 0]))
        radius_line = Line(wheel.get_center(), wheel.get_center() + np.array([wheel_radius * np.cos(PI / 4), wheel_radius * np.sin(PI / 4), 0]), color=YELLOW_C, stroke_width=3)
        self.auto_register('radius_line', radius_line)
        radius_label = MathTex('r = 0.35\\text{ m}', font_size=24, color=YELLOW_C)
        self.auto_register('radius_label', radius_label)
        radius_label.next_to(radius_line.get_center(), UP + RIGHT, buff=0.1)
        self.next_section('scene_4_wheel_example_beat_1')
        self.play(FadeIn(header, shift=DOWN * 0.2), Create(ground), FadeIn(road_surface), Create(ground_ticks), run_time=1.9651)
        self.play(FadeIn(wheel, shift=DOWN * 0.5), run_time=1.9651)
        self.play(Create(radius_line), Write(radius_label), run_time=2.2926)
        self.wait(2.4563)
        self.next_section('scene_4_wheel_example_beat_2')
        self.play(FadeOut(radius_line), FadeOut(radius_label), run_time=0.6158)
        start_pt_marker = Dot(np.array([start_x, ground_y, 0]), radius=0.07, color=YELLOW)
        self.auto_register('start_pt_marker', start_pt_marker)
        self.add(start_pt_marker)
        end_x = start_x + one_rev_dist
        self.auto_register('end_x', end_x)
        path_tracker = ValueTracker(0.0)
        self.auto_register('path_tracker', path_tracker)
        trace_line = always_redraw(lambda: Line(np.array([start_x, ground_y + 0.02, 0]), np.array([start_x + path_tracker.get_value() * one_rev_dist, ground_y + 0.02, 0]), color=YELLOW, stroke_width=4))
        self.auto_register('trace_line', trace_line)
        self.add(trace_line)
        initial_wheel_center = wheel.get_center()
        self.auto_register('initial_wheel_center', initial_wheel_center)

        def wheel_updater(m):
            t = path_tracker.get_value()
            current_x = initial_wheel_center[0] + t * one_rev_dist
            m.move_to(np.array([current_x, ground_y + wheel_radius, 0]))
        wheel.add_updater(wheel_updater)
        self.play(path_tracker.animate.set_value(1.0), Rotate(wheel, -TAU, about_point=wheel.get_center()), rate_func=linear, run_time=3.4897)
        wheel.remove_updater(wheel_updater)
        end_pt_marker = Dot(np.array([end_x, ground_y, 0]), radius=0.07, color=YELLOW)
        self.auto_register('end_pt_marker', end_pt_marker)
        self.play(FadeIn(end_pt_marker), run_time=0.3079)
        rev_brace = BraceBetweenPoints(np.array([start_x, ground_y - 0.05, 0]), np.array([end_x, ground_y - 0.05, 0]), DOWN, buff=0.15, color=YELLOW_C)
        self.auto_register('rev_brace', rev_brace)
        rev_text = MathTex('1\\text{ Turn} = C = 2\\pi r = 2\\pi(0.35) \\approx 2.20\\text{ m}', font_size=26, color=YELLOW_C).next_to(rev_brace, DOWN, buff=0.15)
        self.auto_register('rev_text', rev_text)
        self.play(GrowFromCenter(rev_brace), FadeIn(rev_text, shift=UP * 0.2), run_time=1.4369)
        self.wait(1.5396)
        self.next_section('scene_4_wheel_example_beat_3')
        calc_box = RoundedRectangle(corner_radius=0.2, width=8.5, height=1.8, fill_color='#1E222A', fill_opacity=0.88, stroke_color=BLUE_C, stroke_width=2).move_to(np.array([0, 1.2, 0]))
        self.auto_register('calc_box', calc_box)
        multi_turn_line1 = MathTex('\\text{Total Distance} = (\\text{Number of Turns}) \\times C', font_size=28, color=WHITE)
        self.auto_register('multi_turn_line1', multi_turn_line1)
        multi_turn_line2 = MathTex('10\\text{ Turns} = 10 \\times 2.20\\text{ m} = ', '22.0\\text{ m}', font_size=32)
        self.auto_register('multi_turn_line2', multi_turn_line2)
        multi_turn_line2[0].set_color(WHITE)
        multi_turn_line2[1].set_color(GREEN_B)
        calc_content = VGroup(multi_turn_line1, multi_turn_line2).arrange(DOWN, buff=0.25)
        self.auto_register('calc_content', calc_content)
        calc_content.move_to(calc_box.get_center())
        calc_group = VGroup(calc_box, calc_content)
        self.auto_register('calc_group', calc_group)
        self.play(FadeIn(calc_group, shift=DOWN * 0.3), run_time=1.4801)
        self.play(Indicate(multi_turn_line2[1], scale_factor=1.2, color=GREEN_A), run_time=1.7268)
        self.wait(2.2201)
