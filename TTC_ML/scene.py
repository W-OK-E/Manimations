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
        self.play(FadeOut(Group(en)))
        


        ## Show a bunch of examples of the same:
        orig1 = ImageMobject(horse).scale(0.5)
        fake1 = ImageMobject(zebra).scale(0.5)
        orig1.move_to(LEFT + UP*0.75)
        fake1.move_to(RIGHT + UP*0.75)
        self.play(FadeIn(orig1))
        self.play(FadeIn(fake1))
        self.wait(2)
        img2 = ImageMobject('assets/Examples.jpeg').move_to(DOWN*0.5)
        self.play(FadeIn(img2))
        self.wait(2)


        
