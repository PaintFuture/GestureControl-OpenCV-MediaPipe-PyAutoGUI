"""
Hand gesture control using Keras model for classification.

This script demonstrates how to use a Keras model for gesture classification
with MediaPipe hand detection and PyAutoGUI for mouse control.

Features:
- Low-attention/standby mode for background operation with low CPU usage
- Activation gesture (touching hands together) to wake from standby
- Automatic standby after configurable idle timeout
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import argparse

from keras_gesture_classifier import GestureClassifier

drawing_utils = mp.solutions.drawing_utils
hands_module = mp.solutions.hands

# Global variable for mouse drag state
is_holding = False

# Configuration for action cooldown
ACTION_COOLDOWN = 0.5  # Reduced from 2 seconds for better responsiveness
last_action_time = time.time()

# Standby mode configuration
STANDBY_TIMEOUT = 60  # Seconds of inactivity before entering standby mode
STANDBY_FRAME_INTERVAL = 10  # Process every Nth frame in standby mode (reduces CPU)
ACTIVE_FRAME_INTERVAL = 1  # Process every frame in active mode

# Initialize video capture
capture = cv2.VideoCapture(0)

# Video input's screen dimensions
if capture.isOpened():
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Device's screen dimensions
screen_width, screen_height = pyautogui.size()


def detect_hands_touching(detection_results):
    """
    Detect if two hands are touching (activation gesture for waking from standby).
    
    This checks if both hands are detected and their palm centers are close together.
    
    Args:
        detection_results: MediaPipe hand detection results.
    
    Returns:
        True if hands are detected touching, False otherwise.
    """
    if not detection_results.multi_hand_landmarks:
        return False
    
    if len(detection_results.multi_hand_landmarks) < 2:
        return False
    
    # Get palm center (wrist) for each hand
    hand1_landmarks = detection_results.multi_hand_landmarks[0]
    hand2_landmarks = detection_results.multi_hand_landmarks[1]
    
    # Use wrist position as palm center
    wrist1 = hand1_landmarks.landmark[hands_module.HandLandmark.WRIST]
    wrist2 = hand2_landmarks.landmark[hands_module.HandLandmark.WRIST]
    
    # Calculate distance between wrists
    distance = np.sqrt(
        (wrist1.x - wrist2.x) ** 2 + 
        (wrist1.y - wrist2.y) ** 2
    )
    
    # Threshold for "touching" - hands need to be very close
    TOUCH_THRESHOLD = 0.15
    
    return distance < TOUCH_THRESHOLD


def execute_gesture_action(gesture_name, landmarks):
    """
    Execute the corresponding action for a detected gesture.
    
    Args:
        gesture_name: The name of the detected gesture.
        landmarks: MediaPipe hand landmarks for position information.
    
    Returns:
        The current time if an action was performed, None otherwise.
    """
    global is_holding
    
    # Get palm position for mouse movement (using middle finger base)
    middle_base = landmarks.landmark[hands_module.HandLandmark.MIDDLE_FINGER_MCP]
    
    if gesture_name == "pinch":
        if not is_holding:
            pyautogui.mouseDown(button="left")
            is_holding = True
        return time.time()
    
    elif gesture_name == "left_click":
        if is_holding:
            pyautogui.mouseUp(button="left")
            is_holding = False
        pyautogui.click(button="left")
        return time.time()
    
    elif gesture_name == "right_click":
        if is_holding:
            pyautogui.mouseUp(button="left")
            is_holding = False
        pyautogui.click(button="right")
        return time.time()
    
    elif gesture_name == "scroll_up":
        pyautogui.scroll(3)
        return time.time()
    
    elif gesture_name == "scroll_down":
        pyautogui.scroll(-3)
        return time.time()
    
    else:  # "none" or unrecognized gesture
        if is_holding:
            pyautogui.mouseUp(button="left")
            is_holding = False
        # Move mouse to palm position
        pyautogui.moveTo(
            middle_base.x * screen_width,
            middle_base.y * screen_height
        )
        return None


def main(model_path=None, confidence_threshold=0.5, standby_timeout=60, low_attention_mode=True):
    """
    Main function to run the gesture control with Keras model.
    
    Args:
        model_path: Path to the Keras model file. If None, will raise an error.
        confidence_threshold: Minimum confidence for gesture detection.
        standby_timeout: Seconds of inactivity before entering standby mode.
        low_attention_mode: If True, enables standby mode for low CPU usage.
    """
    global last_action_time
    
    # Initialize the gesture classifier
    if model_path is None:
        print("Error: No model path provided.")
        print("Please provide a trained Keras model using --model argument.")
        print("You can create a sample model structure using:")
        print("  python -c \"from keras_gesture_classifier import create_sample_model; create_sample_model()\"")
        return
    
    try:
        classifier = GestureClassifier(model_path=model_path)
        print(f"Loaded gesture model from {model_path}")
        print(f"Gesture labels: {classifier.gesture_labels}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Standby mode state
    is_standby = False
    last_gesture_time = time.time()
    frame_count = 0
    
    with hands_module.Hands(
        min_detection_confidence=0.8, min_tracking_confidence=0.5
    ) as hands:
        print("Starting gesture control. Press 'q' to quit.")
        if low_attention_mode:
            print(f"Low-attention mode enabled. Standby after {standby_timeout}s of inactivity.")
            print("Touch both hands together to wake from standby.")
        
        while capture.isOpened():
            read_success, frame = capture.read()
            if not read_success:
                continue
            
            frame_count += 1
            
            # In standby mode, only process every Nth frame to reduce CPU usage
            current_frame_interval = STANDBY_FRAME_INTERVAL if is_standby else ACTIVE_FRAME_INTERVAL
            if frame_count % current_frame_interval != 0:
                # Still show the frame but skip processing
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                
                # Show standby indicator
                if is_standby:
                    cv2.putText(
                        image, "STANDBY - Touch hands to wake", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (128, 128, 128), 2
                    )
                
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Hand Gesture Control (Keras)", image_bgr)
                
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            detection_results = hands.process(image)
            image.flags.writeable = True
            
            # Check for standby timeout
            if low_attention_mode:
                time_since_gesture = time.time() - last_gesture_time
                
                if is_standby:
                    # In standby mode, check for activation gesture (hands touching)
                    if detect_hands_touching(detection_results):
                        is_standby = False
                        last_gesture_time = time.time()
                        print("Waking from standby - hands detected touching!")
                    
                    # Show standby indicator
                    cv2.putText(
                        image, "STANDBY - Touch hands to wake", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (128, 128, 128), 2
                    )
                else:
                    # Check if we should enter standby
                    if time_since_gesture > standby_timeout:
                        is_standby = True
                        print(f"Entering standby mode after {standby_timeout}s of inactivity...")

            if detection_results.multi_hand_landmarks and not is_standby:
                for landmarks in detection_results.multi_hand_landmarks:
                    # Draw landmarks on the image
                    drawing_utils.draw_landmarks(
                        image=image,
                        landmark_list=landmarks,
                        connections=hands_module.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=15
                        ),
                        connection_drawing_spec=drawing_utils.DrawingSpec(
                            color=(255, 255, 0), thickness=10
                        ),
                    )
                    
                    # Use Keras model for gesture prediction
                    gesture_name, confidence = classifier.predict(
                        landmarks, 
                        confidence_threshold=confidence_threshold
                    )
                    
                    # Display gesture and confidence on the image
                    text = f"{gesture_name}: {confidence:.2f}"
                    cv2.putText(
                        image, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2
                    )
                    
                    # Update last gesture time if a meaningful gesture was detected
                    if gesture_name != "none":
                        last_gesture_time = time.time()
                    
                    # Execute gesture action with cooldown
                    if time.time() - last_action_time > ACTION_COOLDOWN:
                        result = execute_gesture_action(gesture_name, landmarks)
                        if result:
                            last_action_time = result

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Hand Gesture Control (Keras)", image_bgr)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hand gesture control using Keras model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained Keras model file (.h5 or SavedModel directory)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for gesture detection (0.0-1.0)"
    )
    parser.add_argument(
        "--standby-timeout",
        type=int,
        default=60,
        help="Seconds of inactivity before entering standby mode (default: 60)"
    )
    parser.add_argument(
        "--no-low-attention",
        action="store_true",
        help="Disable low-attention/standby mode (always run at full CPU)"
    )
    
    args = parser.parse_args()
    main(
        model_path=args.model, 
        confidence_threshold=args.confidence,
        standby_timeout=args.standby_timeout,
        low_attention_mode=not args.no_low_attention
    )
