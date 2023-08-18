import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np
from pynput import keyboard
import pybullet as p
import pybullet_data
import threading

# Configure the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Create alignment primitive with color as its target stream
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Time tracking for FPS calculation
pTime = 0
cTime = 0

# Switch to start/stop detection and initial depth
is_detecting = False
initial_depth = None

# Flags to track fist gesture and depth difference
fist = 0
depth_diff = 0

status = "wait"  # Program starts in waiting state

# Variables to track the detection start and exit
start = 0
end = 0

# Flags for possible hand gestures
up = 0
down = 0
left = 0
right = 0
forward = 0
backward = 0


# Callback function to handle key presses
def on_press(key):
    global start, end
    try:
        if key == keyboard.Key.space and start == 0:
            start = 1
            print("Started detecting")
        elif key == keyboard.Key.space and start == 1:
            start = 0
            reset()
            print("Stopped detecting")
        if key.char == 'q':
            end = 1
            print("Program exit")
    except AttributeError:
        pass


# Set up a callback for key presses
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Define the target position and speed
target_pos = [0, 0, 0.5]
last_position = [0, 0, 0.5]
speed = 0.005


# Function to update target position based on direction
def update_position():
    if up:
        target_pos[2] += speed
    elif down:
        target_pos[2] -= speed
    elif left:
        target_pos[1] -= speed
    elif right:
        target_pos[1] += speed
    elif forward:
        target_pos[0] += speed
    elif backward:
        target_pos[0] -= speed
    time.sleep(1)


# Start the physics simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a plane and the Kuka arm
p.loadURDF("plane.urdf")
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

# Get the ID of the end-effector link
endEffectorId = 6
# endEffectorId = 11

# Create a list to store the previous end-effector position
prev_position = None


def get_end_effector_position(robot):
    # The link id for the end effector depends on the robot
    state = p.getLinkState(robot, endEffectorId)
    return state[0]  # position


# Function to move the arm
def move_arm():
    global prev_position
    iteration_count = 0
    while True:
        p.stepSimulation()
        jointPoses = p.calculateInverseKinematics(kukaId, endEffectorId, target_pos)
        for i in range(7):
            p.setJointMotorControl2(kukaId, i, p.POSITION_CONTROL, jointPoses[i])
        # Get the current position of the end effector
        current_position = p.getLinkState(kukaId, endEffectorId)[0]

        # Format the position to three decimal places
        formatted_position = tuple([round(coord, 3) for coord in current_position])

        print(f"End Effector Position: {formatted_position}")

        # If the previous position is not None, draw a line from the previous position to the current position
        if prev_position is not None and iteration_count > 50:
            p.addUserDebugLine(prev_position, current_position, [1, 1, 0], 2)
        iteration_count += 1
        # Update the previous position
        prev_position = current_position
        time.sleep(1. / 240.)


# Reset the gesture flags to default
def reset():
    global up, down, left, right, forward, backward
    up = 0
    down = 0
    left = 0
    right = 0
    forward = 0
    backward = 0


# Start the move_arm function in a separate thread
threading.Thread(target=move_arm).start()

while True:
    threading.Thread(target=update_position).start()

    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image = cv2.flip(depth_image, 1)
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.flip(color_image, 1)

    h, w, c = color_image.shape

    if start:
        # Process image through MediaPipe Hands
        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_index, handHns in enumerate(results.multi_handedness):
                label = handHns.classification[0].label
                if label == "Left":
                    handLms = results.multi_hand_landmarks[hand_index]
                    # Get depth of the wrist (reference point)
                    wrist_x = min(int(handLms.landmark[0].x * 640), 639)
                    wrist_y = min(int(handLms.landmark[0].y * 480), 479)

                    wrist_depth = depth_image[wrist_y, wrist_x]

                    print('Current depth: ', wrist_depth)

                    # If this is the first frame after we started detecting, store the initial depth
                    if initial_depth is None and fist:
                        initial_depth = wrist_depth

                    if fist == 0:
                        initial_depth = None

                    if initial_depth is not None:
                        # Calculate distance from the initial depth

                        if initial_depth < wrist_depth:
                            print('The currently recognised direction signal is: BACKWARD')
                            reset()
                            backward = 1

                        if initial_depth > wrist_depth and handLms.landmark[4].y < handLms.landmark[0].y:
                            print('The currently recognised direction signal is: FORWARD')
                            reset()
                            forward = 1

                    index_tip = handLms.landmark[8]
                    middle_tip = handLms.landmark[12]
                    ring_tip = handLms.landmark[16]
                    pinky_tip = handLms.landmark[20]

                    index_pip = handLms.landmark[6]
                    middle_pip = handLms.landmark[10]
                    ring_pip = handLms.landmark[14]
                    pinky_pip = handLms.landmark[18]

                    # Check if four fingers are bent
                    if (index_tip.y >= index_pip.y and
                        middle_tip.y >= middle_pip.y and
                        ring_tip.y >= ring_pip.y and
                        pinky_tip.y >= pinky_pip.y) and \
                            (handLms.landmark[4].y < handLms.landmark[0].y):
                        fist = 1
                    else:
                        fist = 0

                    if fist == 0:
                        # processing logic...
                        for id, lm in enumerate(handLms.landmark):

                            cx, cy = int(lm.x * w), int(lm.y * h)

                            if id == 0:
                                wrist_x = cx
                                wrist_y = cy
                            if id == 4:
                                thumb_x = cx
                                thumb_y = cy
                            if id == 8:
                                index_x = cx
                                index_y = cy
                            if id == 12:
                                middle_x = cx
                                middle_y = cy
                            if id == 16:
                                ring_x = cx
                                ring_y = cy
                            if id == 20:
                                pinky_x = cx
                                pinky_y = cy

                        if (((thumb_y - wrist_y) < 0) and ((middle_y - thumb_y) < 0)) and (index_y - wrist_y) < 0 and \
                                (middle_y - wrist_y) < 0 and (ring_y - wrist_y) < 0 and (pinky_y - wrist_y) < 0:
                            print('The currently recognised direction signal is:  UP')
                            reset()
                            up = 1
                        if (((thumb_y - wrist_y) > 0) and ((middle_y - thumb_y) > 0)) and (index_y - wrist_y) > 0 and \
                                (middle_y - wrist_y) > 0 and (ring_y - wrist_y) > 0 and (pinky_y - wrist_y) > 0 and \
                                (thumb_x > middle_x):
                            print('The currently recognised direction signal is: DOWN')
                            reset()
                            down = 1
                        if (thumb_x - wrist_x) < 0 and (index_x - wrist_x) < 0 and (middle_x - wrist_x) < 0 and \
                                (ring_x - wrist_x) < 0 and (pinky_x - wrist_x) < 0:
                            print('The currently recognised direction signal is: LEFT')
                            reset()
                            left = 1
                        if (thumb_x - wrist_x) > 0 and (index_x - wrist_x) > 0 and   (middle_x - wrist_x) > 0 and \
                                (ring_x - wrist_x) > 0 and (pinky_x - wrist_x) > 0:
                            print('The currently recognised direction signal is: RIGHT')
                            reset()
                            right = 1

                    # Draw the hand landmarks
                    mpDraw.draw_landmarks(color_image, handLms, mpHands.HAND_CONNECTIONS)
        else:
            print('No gestures are currently recognised!')

    # FPS Calculation and Display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(color_image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("HandsImage", color_image)
    cv2.waitKey(1)

    if end == 1:
        break

    time.sleep(0.05)

pipeline.stop()
