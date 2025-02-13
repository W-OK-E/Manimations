from manim import *

img1_path = "assets/Horse.png"
img2_path = "assets/Zebra.png"
class DissolveAnimation(Scene):
    def construct(self):
        # Load the two images
        image1 = ImageMobject("assets/Horse.png").scale()  # Scale to desired size
        image2 = ImageMobject("assets/Zebra.png").scale(3)  # Scale to desired size

        # Place the images at the center
        image1.move_to(ORIGIN)
        image2.move_to(ORIGIN)

        # Display the first image
        self.add(image1)
        self.wait(1)

        # Create dissolve effect
        # Apply VGroup for tiny squares that transition one image into the other
        grid_size = 20
        squares = VGroup()
        width, height = image1.width, image1.height

        for i in range(grid_size):
            for j in range(grid_size):
                x = -width / 2 + i * (width / grid_size)
                y = -height / 2 + j * (height / grid_size)
                square = Square(
                    side_length=width / grid_size,
                    fill_opacity=1,
                    fill_color=BLACK,
                    stroke_width=0,
                ).move_to([x, y, 0])
                squares.add(square)

        # Fade in the grid of squares
        self.play(FadeIn(squares, run_time=2))

        # Replace the image while dissolving squares
        self.remove(image1)
        self.add(image2)

        # Dissolve out the squares to reveal the second image
        self.play(FadeOut(squares, run_time=2))
        self.wait(1)




# Create a custom scene for the animation
class ImageToModel(Scene):
    def construct(self):
        # Load and display the image
        image = ImageMobject(img1_path)
        image.scale(0.5)
        image.move_to(ORIGIN)
        self.play(FadeIn(image))

        # Create a rectangle to represent the deep learning model
        model = Rectangle(width=4, height=2, stroke_color=WHITE)
        model.next_to(image, DOWN, buff=0.5)
        model_label = Text("Deep Learning Model")
        model_label.next_to(model, DOWN)

        # Arrow to represent the image input to the model
        arrow = Arrow(image.get_bottom(), model.get_top(), buff=0.1)
        arrow_label = Text("Input Image")
        arrow_label.next_to(arrow, UP)
        self.play(FadeOut(image))
        # Animate the image moving to the model
        #self.play(Transform(image.copy(), model))
        self.play(Create(arrow), Write(arrow_label))
        self.play(Write(model_label))
        self.wait(2)
        """
        # Fade out everything at the end
        self.play(FadeOut(image), FadeOut(arrow), FadeOut(arrow_label), FadeOut(model), FadeOut(model_label))"""




from manim import *
from PIL import Image
from manim_ml.neural_network import Convolutional2DLayer, FeedForwardLayer, NeuralNetwork,ImageLayer, ImageToConvolutional2DLayer
from manim_ml.neural_network import Convolutional2DToFeedForward
from manim_ml.neural_network import FeedForwardToImage
# This changes the resolution of our rendered videos
config.pixel_height = 700
config.pixel_width = 1900
config.frame_height = 7.0
config.frame_width = 7.0

# Here we define our basic scene
class BasicScene(ThreeDScene):

    # The code for generating our scene goes here
    def construct(self):
        horse = Image.open(img1_path)
        zebra = Image.open(img2_path) # You will need to download an image of a digit.
        horse = np.asarray(horse)
        zebra = np.asarray(zebra)
        text=Tex("Style Transfer")
        text.font_size= 15
        self.play(Write(text,run_time = 1))
        self.play(FadeOut(text))
        # self.play(text.animate.move_to(UP+LEFT))

        im_layer_1=ImageLayer(horse,height=1.5,show_image_on_create=True)
        im_layer_2=ImageLayer(zebra,height=1.5,show_image_on_create=False)
        cv_layer_1=Convolutional2DLayer(1, 7, 2, filter_spacing=0.32)
        cv_layer_2=Convolutional2DLayer(1, 5, 3, filter_spacing=0.18)
        fd1 =   FeedForwardLayer(6)
        fd2 = FeedForwardLayer(5)
        fd_2_img = FeedForwardToImage(fd2,im_layer_2)
        en = NeuralNetwork([
                im_layer_1,
                cv_layer_1, # Note the default stride is 1.
                cv_layer_2,
                # Convolutional2DLayer(5, 3, 3, filter_spacing=0.18),
                fd1,
                fd2,
                im_layer_2,
            ],
            layer_spacing=0.30,
        )
        
        self.play(FadeIn(en))

        self.play(en.animate.to_edge(LEFT))

        forward_pass = en.make_forward_pass_animation(run_time=10.0)
        self.play(forward_pass)

        self.wait(2)
        # self.play(FadeOut(Group(en)))
        self.clear()
        self.wait(1)


        ## Show a bunch of examples of the same:
        # orig1 = ImageMobject(horse).scale(0.5)
        # fake1 = ImageMobject(zebra).scale(0.5)
        # orig1.move_to(LEFT + UP*0.75)
        # fake1.move_to(RIGHT + UP*0.75)
        # self.play(FadeIn(orig1))
        # self.play(FadeIn(fake1))
        # self.wait(2)
        # img2 = ImageMobject('assets/Examples.jpeg').move_to(DOWN*0.5)
        # self.play(FadeIn(img2))
        # self.wait(2)
        


        text = Text("Examples of Image to Image Translation")
        text.font_size = 15
        self.play(Write(text))
        self.play(FadeOut(text))
        sum2wint = ImageMobject('assets/sum2wint.png').scale(0.4).move_to(LEFT*1.5 + UP*0.5)
        self.play(Write(Text("Summer Landscape to\n   Winter Landscape").scale(0.2).move_to(LEFT*1.5+DOWN)))
        self.play(FadeIn(sum2wint))
        photo2paint = ImageMobject('assets/photo2paint.png').scale(0.4).move_to(RIGHT*1.5 + UP*0.5)
        self.play(Write(Text("PhotoGraphs to Paintings").scale(0.2).move_to(RIGHT*1.5 + DOWN)))
        self.play(FadeIn(photo2paint))

        self.wait(2.0)

        self.play(FadeOut(sum2wint),FadeOut(photo2paint))
        self.clear()
        self.wait(1)

        text = Text("But how to do it?").scale(0.4).move_to(UP)
        think = ImageMobject("assets/thinker.png").scale(0.5).move_to(DOWN*0.5)
        self.add(think)
        self.add(text)

        self.wait(2)


        #This section demonstrates the requirement for pair-wise iamges for style transfer
        # Black and White and Color Image is taken
        #############################################################333
        img1 = ImageMobject("assets/bw.png").scale(1).to_edge(LEFT)
        img2 = ImageMobject("assets/color.png").scale(1).to_edge(RIGHT)

        # Model Box
        model_box = Rectangle(width=1, height=1, color=WHITE)
        model_label = Text("Model").scale(0.3).move_to(model_box.get_center())
       
        # Position model box in center
        model_group = VGroup(model_box, model_label)
        model_group.move_to(ORIGIN)

        # Add initial image and model box
        self.play(FadeIn(img1), FadeIn(img2))
        self.wait(1)
        self.play(FadeIn(model_group))

        # Warp effect on Image 1 before entering the model
        self.play(img1.animate.scale(0.1).move_to(model_box.get_center()),
                  run_time=1.5)
        
  
        # Make Image 1 disappear as it enters the model
        self.play(FadeOut(img1))
        self.wait(0.5)
        
        self.play(model_group.animate.move_to(LEFT*2))
        output_label = Text("Output").scale(0.1).move_to(model_box.get_center())
        self.play(output_label.animate.scale(3).move_to(RIGHT*0.5))

        self.wait(2)

        # Fade out everything
        self.play(FadeOut(img2, model_group,output_label))
        ###############################################################


        #This section is a brief intro to Generator and Discriminator Networks
        text = Text("Cycle GANs")
        text = text.scale(0.5)

        self.play(Write(text))
        self.wait(1)
        self.play(FadeOut(text))


        text = Text("Generators")
        text = text.scale(0.5).move_to(UP)
        self.play(Write(text))
        




class TestScene(ThreeDScene):
    def construct(self):
        img = ImageMobject("assets/color.png").to_edge(LEFT)
        # Create fine grid
        grid = [
            Line(start, end, stroke_width=0.5, color=GRAY)
            for x in np.linspace(-3, -1.5, 15) for start, end in [
                ([x, -1, 0], [x, 1, 0])
            ]
        ]
        grid.extend(
            [
            Line(start, end, stroke_width=0.5, color=GRAY)
            for x in np.linspace(-3, -1, 15) for start, end in [
                ([-3, x+2, 0], [-1.5, x+2, 0])
            ]
        ])

        fine_grid = VGroup(*grid)
        fine_grid.set_opacity(0)  # Initially invisible

        # Smaller rectangular latent space grid
        latent_grid = VGroup(*[
            Line(start, end, stroke_width=1.5, color=WHITE)
            for x in np.linspace(-1.5, 1.5, 6) for start, end in [
                ([-1.5, x, 0], [1.5, x, 0]),
                ([x, -1.5, 0], [x, 1.5, 0])
            ]
        ])
        latent_grid.move_to(RIGHT * 1).scale(0.6).set_opacity(1)  # Start hidden

        # Latent space label
        latent_text = Text("Latent Space", font_size=14).next_to(latent_grid, DOWN).set_opacity(0)

        # Show image
        self.play(FadeIn(img))
        self.wait(1)

        # Fade out image while fading in the fine grid
        self.play(FadeOut(img), fine_grid.animate.set_opacity(1), run_time=1.5)
        self.wait(1)

        # Transform fine grid into latent space grid
        self.play(Transform(fine_grid, latent_grid), run_time=1.5)
        self.wait(0.5)

        # Show latent space label
        self.play(FadeIn(latent_text))
        self.wait(2)

        # Fade everything out
        self.play(FadeOut(latent_grid, latent_text))