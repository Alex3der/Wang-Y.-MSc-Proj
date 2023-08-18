# Gesture Recognition in Human-Computer Interaction

This project is an application of gesture recognition in human-computer interaction implemented using Intel RealSense D455 depth camera, MediaPipe Hands and PyBullet. The main function is to control the robotic arm in the simulation environment through gestures to perform the movement corresponding to the gestures.

## Requirements
This project requires Python 3.8 and the following Python libraries installed:
- mediapipe
- cv2
- pyrealsense2
- numpy
- pynput
- pybullet
- pybullet_data
- threading
- time

## Code
`sc223yw.py`: This is the main file that you would want to run. It initializes the parameters, runs the system, and generates the output.

## Run
To run the code, navigate to the project directory and type the following command:
- python sc223yw.py

## How it Works
The system will initialise an image acquisition window returned by the camera.

When the space bar on the keyboard is hit, the gesture detection function is enabled.

The system will initialise a PyBullet virtual environment and a kuka robotic arm. When the gesture is successfully detected and translated into a direction signal, the robotic arm will be controlled to move in the corresponding direction according to the direction signal.

The system will automatically exit when the 'q' key on the keyboard is struck.

## Output
After the gesture is detected, the system will display the image with the Landmark Model attached in the image acquisition window, and output the direction signal corresponding to the currently recognised gesture, as well as the depth information of the wrist area in the console.

The arm will draw a trajectory during the movement and continuously output the coordinates of the end of the arm.

## Customization
It is possible to change the correspondence between gestures and direction signals in the logic module, or add custom gesture logic decisions and corresponding signals.

You can change the type of arm, the speed of movement, and the mode of movement.
