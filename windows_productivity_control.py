"""
Hand Gesture Control for Windows Productivity
Static gesture controls with left/right hand direction for switching windows, tabs, and virtual desktops.
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

print("All imports successful!")
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")

drawing_utils = mp.solutions.drawing_utils
hands_module = mp.solutions.hands

# Configuration
ACTION_COOLDOWN = 1.0  # Deadtime after gesture execution
STABLE_DURATION = 0.5  # Gesture must be stable for this duration before triggering

# State tracking
last_action_time = 0  # Last time an action was executed
gesture_start_time = 0  # When current stable gesture started
stable_gesture = "none"  # Currently stable gesture
stable_hand = "none"  # Currently stable hand
last_gesture_name = "none"
last_hand_label = "none"

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


def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks."""
    return np.sqrt(
        (landmark1.x - landmark2.x) ** 2 + 
        (landmark1.y - landmark2.y) ** 2 + 
        (landmark1.z - landmark2.z) ** 2
    )


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


def detect_gesture(landmarks):
    """
    Detect static hand gestures based on finger positions.
    
    Returns: gesture name ('pinch', 'fist', 'fox', 'none')
    """
    count, finger_states = count_extended_fingers(landmarks)
    
    # Get landmarks for distance checks
    thumb_tip = landmarks.landmark[hands_module.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[hands_module.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[hands_module.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[hands_module.HandLandmark.RING_FINGER_TIP]
    
    # PINCH: Thumb and index finger touching (close together)
    thumb_index_dist = calculate_distance(thumb_tip, index_tip)
    if thumb_index_dist < 0.05:  # Very close together
        return 'pinch'
    
    # FOX: Ring and middle touching thumb (all three close together)
    # Check if ring and middle are NOT extended and close to thumb
    thumb_middle_dist = calculate_distance(thumb_tip, middle_tip)
    thumb_ring_dist = calculate_distance(thumb_tip, ring_tip)
    
    if (not finger_states['middle'] and not finger_states['ring'] and 
        thumb_middle_dist < 0.08 and thumb_ring_dist < 0.08):
        return 'fox'
    
    # FIST: All fingers closed (no fingers extended)
    if count == 0:
        return 'fist'
    
    return 'none'


def check_and_execute_action(gesture_name, hand_label):
    """
    Execute action only if gesture has been stable for STABLE_DURATION
    and we're not in cooldown period.
    
    Hand direction determines forward/backward:
    - Right hand = forward (next window/tab/desktop)
    - Left hand = backward (previous window/tab/desktop)
    """
    global last_action_time, last_gesture_name, last_hand_label
    global gesture_start_time, stable_gesture, stable_hand
    
    current_time = time.time()
    
    # Check if we're in cooldown period (no events allowed)
    if current_time - last_action_time < ACTION_COOLDOWN:
        # Reset stability tracking during cooldown
        stable_gesture = "none"
        stable_hand = "none"
        gesture_start_time = 0
        return None, 0
    
    # Check if gesture/hand changed
    if gesture_name != stable_gesture or hand_label != stable_hand:
        # New gesture detected, start tracking stability
        stable_gesture = gesture_name
        stable_hand = hand_label
        gesture_start_time = current_time
        return None, 0
    
    # Same gesture continues - check if it's been stable long enough
    if gesture_name == 'none' or hand_label == 'none':
        return None, 0
    
    stability_duration = current_time - gesture_start_time
    
    # Execute action if stable for required duration
    if stability_duration >= STABLE_DURATION:
        action_performed = None
        
        # FIST = Window switching (Alt+Tab)
        if gesture_name == 'fist':
            if hand_label == "Right":
                pyautogui.hotkey('alt', 'tab')
                action_performed = "Fist (Right) -> Next Window"
            else:  # Left hand
                pyautogui.hotkey('alt', 'shift', 'tab')
                action_performed = "Fist (Left) -> Previous Window"
        
        # PINCH = Tab switching (Ctrl+Tab)
        elif gesture_name == 'pinch':
            if hand_label == "Right":
                pyautogui.hotkey('ctrl', 'tab')
                action_performed = "Pinch (Right) -> Next Tab"
            else:  # Left hand
                pyautogui.hotkey('ctrl', 'shift', 'tab')
                action_performed = "Pinch (Left) -> Previous Tab"
        
        # FOX = Virtual Desktop switching (Ctrl+Win+Arrow)
        elif gesture_name == 'fox':
            if hand_label == "Right":
                pyautogui.hotkey('ctrl', 'win', 'right')
                action_performed = "Fox (Right) -> Next Virtual Desktop"
            else:  # Left hand
                pyautogui.hotkey('ctrl', 'win', 'left')
                action_performed = "Fox (Left) -> Previous Virtual Desktop"
        
        if action_performed:
            last_action_time = current_time
            last_gesture_name = gesture_name
            last_hand_label = hand_label
            # Reset stability tracking
            stable_gesture = "none"
            stable_hand = "none"
            gesture_start_time = 0
            print(f"[OK] {action_performed}")
            return action_performed, 0
    
    return None, stability_duration


# Main loop
print("\n=== Hand Gesture Control Started ===")
print("\n[*] Static Gesture Controls with Hand Direction:")
print("\n  FIST (all fingers closed):")
print("    Right hand: Next Window (Alt+Tab)")
print("    Left hand: Previous Window (Alt+Shift+Tab)")
print("\n  PINCH (thumb + index touching):")
print("    Right hand: Next Tab (Ctrl+Tab)")
print("    Left hand: Previous Tab (Ctrl+Shift+Tab)")
print("\n  FOX (ring + middle touching thumb):")
print("    Right hand: Next Virtual Desktop (Ctrl+Win+Right)")
print("    Left hand: Previous Virtual Desktop (Ctrl+Win+Left)")
print("\n[i] Hold gesture stable for 0.5 second to trigger")
print("[i] 1 second deadtime after each action")
print("Press 'q' in the video window to quit.\n")

with hands_module.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1  # Track one hand at a time
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
        hand_label = "none"
        
        if detection_results.multi_hand_landmarks and detection_results.multi_handedness:
            for landmarks, handedness in zip(detection_results.multi_hand_landmarks, 
                                              detection_results.multi_handedness):
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label
                
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
                
                # Detect static gesture
                current_gesture = detect_gesture(landmarks)
                
                # Check stability and execute if ready
                action, stability_time = check_and_execute_action(current_gesture, hand_label)
        
        # Display current gesture and hand on screen
        gesture_labels = {
            'pinch': 'Pinch',
            'fist': 'Fist',
            'fox': 'Fox',
            'none': ''
        }
        
        display_text = ""
        if current_gesture != 'none' and hand_label != 'none':
            display_text = f"{gesture_labels.get(current_gesture)} ({hand_label})"
        
        # Calculate stability progress
        current_time = time.time()
        time_since_action = current_time - last_action_time
        stability_progress = 0
        
        if stable_gesture != 'none' and gesture_start_time > 0:
            stability_progress = current_time - gesture_start_time
        
        if display_text:
            # Color based on state
            in_cooldown = time_since_action < ACTION_COOLDOWN
            
            if in_cooldown:
                color = (128, 128, 128)  # Gray during cooldown
            elif stability_progress >= STABLE_DURATION:
                color = (0, 255, 0)  # Green when ready to trigger
            else:
                color = (255, 255, 0)  # Yellow while building up
            
            cv2.putText(
                image, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2
            )
        
        # Show cooldown timer or stability progress
        if time_since_action < ACTION_COOLDOWN:
            remaining = ACTION_COOLDOWN - time_since_action
            cv2.putText(
                image, f"Cooldown: {remaining:.1f}s", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (128, 128, 128), 1
            )
        elif stability_progress > 0 and stability_progress < STABLE_DURATION:
            remaining_stability = STABLE_DURATION - stability_progress
            cv2.putText(
                image, f"Hold: {remaining_stability:.1f}s", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 0), 1
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
print("\n[OK] Gesture control stopped.")
