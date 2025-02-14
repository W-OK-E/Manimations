## To run the file first setup the environment, Run:
``conda env create --file environment.yaml``
## Or first create the environment and then install the requirements:
``pip install requirements.txt``

Activate the envionment and run:
``manim -pqh scene.py BasicScene``

The flags -pqh specify that the end result will be played at high quality, you can decide not to play it, but the
q flag controls the quality regardless


## Animation Rundown
First a General Description of the Bigger Problem Statement -  Style Transfer 
Then to discuss two ways of doing it - CNNs and Cycle GANs.
Then a detour into CNNs.