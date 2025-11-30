"""
Collect training data for gesture classification.

This script helps you record hand landmarks for different gestures
to train your Keras model.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
from datetime import datetime

drawing_utils = mp.solutions.drawing_utils
hands_module = mp.solutions.hands

# Gesture labels to collect
GESTURES = ["none", "pinch", "left_click", "right_click", "scroll_up", "scroll_down"]

def collect_data():
    """Collect training data for each gesture."""
    
    # Storage for collected data
    training_data = {gesture: [] for gesture in GESTURES}
    
    print("=== Gesture Training Data Collector ===")
    print("\nGestures to collect:")
    for i, gesture in enumerate(GESTURES):
        print(f"  {i}: {gesture}")
    print("\nInstructions:")
    print("  - Press number key (0-5) to select gesture to record")
    print("  - Press SPACE to capture current hand position")
    print("  - Press 's' to save all collected data")
    print("  - Press 'q' to quit without saving")
    print("\nRecommended: Collect 50-100 samples per gesture from different angles/positions\n")
    
    current_gesture_idx = 0
    current_gesture = GESTURES[current_gesture_idx]
    
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not capture.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    with hands_module.Hands(
        min_detection_confidence=0.8, min_tracking_confidence=0.5
    ) as hands:
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            
            # Draw on image
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
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
            
            # Display current gesture and sample count
            text = f"Recording: {current_gesture} (Samples: {len(training_data[current_gesture])})"
            cv2.putText(
                image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )
            
            # Display instructions
            cv2.putText(
                image, "SPACE: Capture | 0-5: Select gesture | S: Save | Q: Quit",
                (10, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )
            
            # Display total samples
            total_samples = sum(len(samples) for samples in training_data.values())
            cv2.putText(
                image, f"Total samples: {total_samples}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2
            )
            
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Data Collection", image_bgr)
            
            key = cv2.waitKey(5) & 0xFF
            
            # Quit
            if key == ord('q'):
                print("\nQuitting without saving...")
                break
            
            # Save data
            elif key == ord('s'):
                save_training_data(training_data)
                break
            
            # Select gesture (0-5)
            elif key >= ord('0') and key <= ord('5'):
                idx = key - ord('0')
                if idx < len(GESTURES):
                    current_gesture_idx = idx
                    current_gesture = GESTURES[current_gesture_idx]
                    print(f"Switched to recording: {current_gesture}")
            
            # Capture sample
            elif key == ord(' '):
                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]
                    
                    # Extract landmarks as list
                    coords = []
                    for lm in landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    
                    training_data[current_gesture].append(coords)
                    print(f"Captured sample #{len(training_data[current_gesture])} for {current_gesture}")
                else:
                    print("No hand detected! Please show your hand to the camera.")
    
    capture.release()
    cv2.destroyAllWindows()


def save_training_data(training_data):
    """Save collected training data to file."""
    
    # Check if we have any data
    total_samples = sum(len(samples) for samples in training_data.values())
    if total_samples == 0:
        print("No data collected. Not saving.")
        return
    
    # Create data directory if it doesn't exist
    os.makedirs("training_data", exist_ok=True)
    
    # Save as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_data/gesture_data_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n✓ Saved {total_samples} samples to {filename}")
    print("\nSamples per gesture:")
    for gesture, samples in training_data.items():
        print(f"  {gesture}: {len(samples)}")
    
    print("\nNext steps:")
    print(f"  1. Run: python train_model.py {filename}")
    print("  2. Use trained model: python keras_gesture_control.py --model trained_gesture_model.h5")


if __name__ == "__main__":
    collect_data()
