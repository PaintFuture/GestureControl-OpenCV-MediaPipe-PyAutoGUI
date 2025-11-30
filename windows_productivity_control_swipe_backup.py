"""
Hand Gesture Control for Windows Productivity
Swipe and gesture controls for switching windows, tabs, and virtual desktops.
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
from collections import deque

print("All imports successful!")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")

drawing_utils = mp.solutions.drawing_utils
hands_module = mp.solutions.hands

# Configuration
ACTION_COOLDOWN = 1.0  # Deadtime between gestures
SWIPE_THRESHOLD = 0.08  # Minimum distance for swipe (smaller = easier)
SWIPE_FRAMES = 5  # Number of frames to detect swipe
GESTURE_LOCK_TIME = 0.5  # Seconds to "lock" gesture after detection

# State tracking
last_action_time = time.time()
hand_positions = deque(maxlen=SWIPE_FRAMES)  # Track positions for swipe
last_gesture_name = "none"
locked_gesture = "none"  # Locked gesture for swiping
gesture_lock_time = 0  # When gesture was locked

print("Initializing camera...")
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not capture.isOpened():
    print("ERROR: Could not open camera!")
    print("Please check:")
    print("1. Camera is connected")
    print("2. Camera permissions are granted")
    print("3. No other application is using the camera")
    exit(1)

print("Camera opened successfully!")

if capture.isOpened():
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {width}x{height}")

screen_width, screen_height = pyautogui.size()


def count_extended_fingers(landmarks):
    """
    Count how many fingers are extended.
    
    Returns: (count, finger_states)
        count: number of extended fingers
        finger_states: dict with True/False for each finger
    """
    finger_tips = [
        hands_module.HandLandmark.THUMB_TIP,
        hands_module.HandLandmark.INDEX_FINGER_TIP,
        hands_module.HandLandmark.MIDDLE_FINGER_TIP,
        hands_module.HandLandmark.RING_FINGER_TIP,
        hands_module.HandLandmark.PINKY_TIP,
    ]
    
    finger_pips = [
        hands_module.HandLandmark.THUMB_IP,
        hands_module.HandLandmark.INDEX_FINGER_PIP,
        hands_module.HandLandmark.MIDDLE_FINGER_PIP,
        hands_module.HandLandmark.RING_FINGER_PIP,
        hands_module.HandLandmark.PINKY_PIP,
    ]
    
    finger_states = {
        'thumb': False,
        'index': False,
        'middle': False,
        'ring': False,
        'pinky': False
    }
    
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
        tip_landmark = landmarks.landmark[tip]
        pip_landmark = landmarks.landmark[pip]
        
        # For thumb, check x-axis (horizontal), for others check y-axis (vertical)
        if i == 0:  # Thumb
            finger_states[finger_names[i]] = tip_landmark.x < pip_landmark.x - 0.05
        else:  # Other fingers
            finger_states[finger_names[i]] = tip_landmark.y < pip_landmark.y
    
    count = sum(finger_states.values())
    return count, finger_states


def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks."""
    return np.sqrt(
        (landmark1.x - landmark2.x) ** 2 + 
        (landmark1.y - landmark2.y) ** 2 + 
        (landmark1.z - landmark2.z) ** 2
    )


def detect_swipe(hand_positions):
    """
    Detect swipe direction based on hand movement.
    
    Returns: swipe direction ('left', 'right', 'up', 'down', or None)
    """
    if len(hand_positions) < 5:
        return None
    
    # Get start and end positions
    start_pos = hand_positions[0]
    end_pos = hand_positions[-1]
    
    # Calculate displacement
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    # Calculate total distance moved
    distance = np.sqrt(dx**2 + dy**2)
    
    # Check if movement is significant enough
    if distance < SWIPE_THRESHOLD:
        return None
    
    # Determine primary direction
    if abs(dx) > abs(dy):
        # Horizontal swipe
        if dx > 0:
            return 'right'
        else:
            return 'left'
    else:
        # Vertical swipe
        if dy > 0:
            return 'down'
        else:
            return 'up'


def detect_gesture(landmarks):
    """
    Detect static hand gestures based on finger positions.
    Only recognizes specific intentional postures.
    
    Returns: gesture name
    """
    count, finger_states = count_extended_fingers(landmarks)
    
    # Pointing: only index extended
    if count == 1 and finger_states['index']:
        return 'point'
    
    # Pinky: only pinky extended
    if count == 1 and finger_states['pinky']:
        return 'pinky'
    
    # Peace sign: only index and middle extended
    if count == 2 and finger_states['index'] and finger_states['middle'] and not finger_states['ring']:
        return 'peace'
    
    return 'none'


def execute_action(gesture_name, swipe_direction):
    """
    Execute Windows keyboard shortcuts based on gesture and swipe.
    
    Uses locked gesture - so even if fingers aren't detected perfectly during swipe,
    the gesture mode is preserved.
    """
    global last_action_time, last_gesture_name
    
    # Check cooldown (deadtime)
    current_time = time.time()
    if current_time - last_action_time < ACTION_COOLDOWN:
        return None
    
    action_performed = None
    
    # Only process swipes when holding a specific gesture (not "none")
    if swipe_direction and gesture_name != 'none':
        
        # Peace sign + swipe = Window switching
        if gesture_name == 'peace':
            if swipe_direction in ['left', 'up']:
                pyautogui.hotkey('alt', 'shift', 'tab')
                action_performed = "Peace (2 fingers) + Swipe <- -> Previous Window"
            elif swipe_direction in ['right', 'down']:
                pyautogui.hotkey('alt', 'tab')
                action_performed = "Peace (2 fingers) + Swipe -> -> Next Window"
        
        # Pointing finger + swipe = Tab switching
        elif gesture_name == 'point':
            if swipe_direction in ['left', 'up']:
                pyautogui.hotkey('ctrl', 'shift', 'tab')
                action_performed = "Point (1 finger) + Swipe <- -> Previous Tab"
            elif swipe_direction in ['right', 'down']:
                pyautogui.hotkey('ctrl', 'tab')
                action_performed = "Point (1 finger) + Swipe -> -> Next Tab"
        
        # Pinky + swipe = Virtual desktop switching
        elif gesture_name == 'pinky':
            if swipe_direction in ['left', 'up']:
                pyautogui.hotkey('ctrl', 'win', 'left')
                action_performed = "Pinky + Swipe <- -> Previous Virtual Desktop"
            elif swipe_direction in ['right', 'down']:
                pyautogui.hotkey('ctrl', 'win', 'right')
                action_performed = "Pinky + Swipe -> -> Next Virtual Desktop"
    
    if action_performed:
        last_action_time = current_time
        last_gesture_name = gesture_name
        print(f"✓ {action_performed}")
        return action_performed
    
    return None


# Main loop
print("\n=== Hand Gesture Control Started ===")
print("\n[*] Simple One-Handed Controls (Hold gesture + small swipe):")
print("\n  Peace (2 fingers: index + middle) + Swipe:")
print("    <-> : Switch Windows")
print("\n  Point (1 finger: index only) + Swipe:")
print("    <-> : Switch Tabs")
print("\n  Pinky (pinky finger only) + Swipe:")
print("    <-> : Switch Virtual Desktops")
print("\n[i] Swipe in any direction (smaller movements work!)")
print("[i] 1 second deadtime between gestures")
print("[i] Gesture locks for 0.5s during swipes")
print("Press 'q' in the video window to quit.\n")

with hands_module.Hands(
    min_detection_confidence=0.7,  # Lower for better detection during movement
    min_tracking_confidence=0.5,
    max_num_hands=1  # Only track one hand for simplicity
) as hands:
    
    while capture.isOpened():
        read_success, frame = capture.read()
        
        if not read_success:
            print("ERROR: Failed to read frame from camera")
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        detection_results = hands.process(image)
        image.flags.writeable = True
        
        current_gesture = "none"
        swipe_direction = None
        
        if detection_results.multi_hand_landmarks:
            for landmarks in detection_results.multi_hand_landmarks:
                # Draw landmarks
                drawing_utils.draw_landmarks(
                    image=image,
                    landmark_list=landmarks,
                    connections=hands_module.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2
                    ),
                    connection_drawing_spec=drawing_utils.DrawingSpec(
                        color=(255, 255, 0), thickness=2
                    ),
                )
                
                # Track hand position for swipe detection
                wrist = landmarks.landmark[hands_module.HandLandmark.WRIST]
                hand_positions.append((wrist.x, wrist.y))
                
                # Detect swipe
                swipe_direction = detect_swipe(hand_positions)
                
                # Detect static gesture
                current_gesture = detect_gesture(landmarks)
                
                # GESTURE LOCKING: Lock in the gesture when first detected
                if current_gesture != 'none':
                    locked_gesture = current_gesture
                    gesture_lock_time = time.time()
                
                # Use locked gesture if still within lock time
                active_gesture = locked_gesture if (time.time() - gesture_lock_time < GESTURE_LOCK_TIME) else current_gesture
                
                # Execute swipe action using the active (possibly locked) gesture
                action = execute_action(active_gesture, swipe_direction)
        else:
            # No hand detected, clear position history
            hand_positions.clear()
            # Reset locked gesture if hand lost for too long
            if time.time() - gesture_lock_time > GESTURE_LOCK_TIME:
                locked_gesture = "none"
        
        # For display, use the active gesture (locked or current)
        display_gesture = locked_gesture if (time.time() - gesture_lock_time < GESTURE_LOCK_TIME) else current_gesture
        
        # Display current gesture on screen
        gesture_labels = {
            'point': 'Point (1)',
            'peace': 'Peace (2)',
            'pinky': 'Pinky',
            'none': ''
        }
        
        display_text = gesture_labels.get(display_gesture, display_gesture)
        
        if swipe_direction and display_gesture != 'none':
            arrows = {'left': '<-', 'right': '->', 'up': '^', 'down': 'v'}
            display_text = f"{gesture_labels.get(display_gesture)} + {arrows.get(swipe_direction, swipe_direction)}"
        
        if display_gesture != 'none':
            # Show if gesture is locked (cyan) vs actively detected (yellow/green)
            is_locked = (locked_gesture != 'none' and time.time() - gesture_lock_time < GESTURE_LOCK_TIME)
            if swipe_direction:
                color = (0, 255, 0)  # Green when swiping
            elif is_locked:
                color = (255, 255, 0)  # Yellow when locked
            else:
                color = (255, 255, 0)  # Yellow when detected
            
            cv2.putText(
                image, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2
            )
            
            # Show lock indicator
            if is_locked and not detection_results.multi_hand_landmarks:
                cv2.putText(
                    image, "[LOCKED]", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1
                )
        
        # Show cooldown timer
        time_since_action = time.time() - last_action_time
        if time_since_action < ACTION_COOLDOWN:
            remaining = ACTION_COOLDOWN - time_since_action
            cv2.putText(
                image, f"Cooldown: {remaining:.1f}s", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (128, 128, 128), 1
            )
        
        # Display instructions
        cv2.putText(
            image, "Press 'q' to quit", (10, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 1
        )
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Gesture Control - Windows Productivity", image_bgr)
        
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
print("\n✓ Gesture control stopped.")
