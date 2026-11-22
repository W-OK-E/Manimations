"""Approved technical smoke test; see DESIGN.md for the storyboard."""

from manim import BLACK, DOWN, WHITE, FadeIn, MathTex, Scene, Text, VGroup


class HelloWorld(Scene):
    def __init__(self, **kwargs):
        # Keep future incidental randomness reproducible too.
        kwargs.setdefault("random_seed", 0)
        super().__init__(**kwargs)

    def construct(self):
        self.camera.background_color = BLACK

        # Use the installed local font, with no external assets.
        title = Text("Hello, Manim!", font="DejaVu Sans", font_size=48, color=WHITE)

        # Compute the displayed result rather than typing an unchecked answer.
        left, right = 1, 1
        total = left + right
        assert total == 2
        formula = MathTex(f"{left} + {right} = {total}", font_size=48, color=WHITE)

        # Grouping keeps both lines centered, with a clear gap between them.
        content = VGroup(title, formula).arrange(DOWN, buff=0.5).move_to([0, 0, 0])
        self.play(FadeIn(content), run_time=1)
        self.wait(3)
