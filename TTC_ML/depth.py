from manim import *
import numpy as np

class BasicScene(Scene):
    def construct(self):
        # Helper function to create input tensor visualization
        def create_tensor(dims, color=BLUE, opacity=0.3, label_text=""):
            tensor = VGroup()
            for z in range(dims[2]):
                rect = Rectangle(
                    width=dims[0]/2,
                    height=dims[1]/2,
                    fill_color=color,
                    fill_opacity=opacity,
                    stroke_color=WHITE
                ).shift(z * 0.2 * RIGHT + z * 0.2 * UP)
                tensor.add(rect)
            
            if label_text:
                label = Text(label_text, font_size=24).next_to(tensor, DOWN)
                return VGroup(tensor, label)
            return tensor

        # Create title
        title = Text("Depthwise Separable Convolution", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Show input tensor
        input_tensor = create_tensor([4, 4, 3], BLUE, 0.3, "Input\n(H×W×C)")
        input_tensor.shift(4*LEFT + UP)
        self.play(Create(input_tensor))
        self.wait()

        # Explain depthwise convolution
        depthwise_title = Text("Step 1: Depthwise Convolution", font_size=36)
        depthwise_title.next_to(title, DOWN)
        self.play(Write(depthwise_title))

        # Show depthwise kernels
        kernels = VGroup()
        kernel_labels = VGroup()
        for i in range(3):
            kernel = Square(side_length=0.5, stroke_color=WHITE)
            kernel.set_fill(RED, opacity=0.3)
            kernel.shift(2*LEFT + (i-1)*UP)
            kernel_label = Text(f"Kernel {i+1}", font_size=20).next_to(kernel, LEFT)
            kernels.add(kernel)
            kernel_labels.add(kernel_label)

        self.play(
            Create(kernels),
            Write(kernel_labels)
        )
        self.wait()

        # Animation for depthwise convolution
        arrows = VGroup()
        for i in range(3):
            arrow = Arrow(
                kernels[i].get_right(),
                input_tensor[0][i].get_left(),
                buff=0.1,
                color=YELLOW
            )
            arrows.add(arrow)

        self.play(Create(arrows))
        self.wait()

        # Show depthwise output
        depthwise_output = create_tensor([4, 4, 3], GREEN, 0.3, "Depthwise\nOutput")
        depthwise_output.shift(RIGHT)
        self.play(Create(depthwise_output))
        self.wait()

        # Explain pointwise convolution
        pointwise_title = Text("Step 2: Pointwise Convolution (1×1 Conv)", font_size=36)
        pointwise_title.next_to(depthwise_title, DOWN)
        self.play(Write(pointwise_title))

        # Show 1x1 convolution kernel
        pointwise_kernel = Square(side_length=0.3, stroke_color=WHITE)
        pointwise_kernel.set_fill(PURPLE, opacity=0.3)
        pointwise_kernel.next_to(depthwise_output, RIGHT, buff=1)
        pointwise_label = Text("1×1 Conv", font_size=20).next_to(pointwise_kernel, UP)

        self.play(
            Create(pointwise_kernel),
            Write(pointwise_label)
        )
        self.wait()

        # Show final output
        final_output = create_tensor([4, 4, 1], YELLOW, 0.3, "Final\nOutput")
        final_output.shift(4*RIGHT + UP)
        self.play(Create(final_output))
        self.wait()

        # Add formula for computational efficiency
        efficiency_text = Text(
            "Computational Efficiency:\nStandard Conv: H×W×Cin×Cout×K×K\nDepthwise Sep Conv: H×W×Cin×K×K + H×W×Cin×Cout",
            font_size=24
        )
        efficiency_text.to_edge(DOWN)
        self.play(Write(efficiency_text))
        self.wait(2)

        # Fade out everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )