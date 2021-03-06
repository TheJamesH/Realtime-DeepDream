# Realtime-DeepDream

Realtime Deepdream combines Google Deepdream and TensorFlow with OpenCV image processing.


- Google DeepDream:https://github.com/google/deepdream
- OpenCV: https://opencv.org/ 


This repository contains scripts for use with a webcam and screen capture.


This is the main loop of the script:

- Run image input frame through Google Deepdream
- While the image is processing morph the current image based on the calculated optical flow of the video
- When the deepdream processing is finished merge the current frame with the new deepdream framed


(note that the deepdream frame also is morphed based on the total optical flow over time since the deepdream process was started)


This script is heavily GPU intensive. Iterations can be kept low in the deepdream process since the frameblending creates a feedback loop. Input with small amounts of motion can build up a heavily processed image over time.


Demonstration: https://youtu.be/FgMyknPBaFo


Controls:

- s - change deepdream layer
- i/k - increase/decrease iterations
- u/j - increase/decrease frame blending factor
- o/l - increase/decrease step size
- q - quit
