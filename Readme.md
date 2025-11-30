# Hand Gesture Control with OpenCV, MediaPipe, and PyAutoGUI

This code demonstrates how to implement hand gesture control using OpenCV, MediaPipe, and PyAutoGUI. It detects hand landmarks, translates them into gestures and performs corresponding actions like clicking and dragging on the computer.

## Requirements

Install all dependencies using:
```bash
pip install -r requirements.txt
```

Or install individually:
- OpenCV: `pip install opencv-python`
- MediaPipe: `pip install mediapipe`
- PyAutoGUI: `pip install pyautogui`
- TensorFlow/Keras (for Keras-based classification): `pip install tensorflow`

## How It Works

### Rule-based Gesture Detection (hand_gesture_control.py)

1. The code initializes a webcam feed using OpenCV.
2. It processes each frame to detect hand landmarks using MediaPipe's hand solutions.
3. The detected landmarks are used to infer hand gestures. The gestures include:
    - Pinch for dragging capabilities.
    - Bending index finger for a left click.
    - Bending middle finger for a right click.
4. PyAutoGUI is used to simulate the mouse actions based on the detected gestures.

### Keras Model-based Gesture Detection (keras_gesture_control.py)

1. Uses the same OpenCV + MediaPipe pipeline for hand detection.
2. Instead of rule-based detection, a trained Keras neural network classifies gestures.
3. Supports custom gesture labels and confidence thresholds.
4. Provides a more flexible and extensible approach for gesture recognition.

To use Keras-based classification:

```bash
# First, create a sample model structure (you'll need to train it with your data)
python -c "from keras_gesture_classifier import create_sample_model; create_sample_model()"

# Then run with your trained model
python keras_gesture_control.py --model gesture_model.h5 --confidence 0.5
```

The `keras_gesture_classifier.py` module provides:
- `GestureClassifier`: A class for loading and using Keras models
- `create_sample_model()`: A helper to create a sample neural network architecture
- `landmarks_to_features()`: Converts MediaPipe landmarks to model input format

## Code Documentation

### Importing Necessary Modules
```python
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
```

### Initializing Necessary Variables and Objects

- `is_holding`: Represents the state of mouse drag.
- `threshold_angle`: Threshold angle for detecting a bent finger. You can adjust this value based on your needs.
- `ACTION_COOLDOWN`: Time interval between each mouse action to avoid spamming.
- `last_action_time`: Keeps track of when the last action was performed.
- `capture`: Video capture object for accessing the webcam.
- `width`, `height`: Dimensions of the webcam feed.
- `screen_width`, `screen_height`: Dimensions of the computer screen.

### Helper Functions

- `calculate_distance(landmark1, landmark2)`: Calculates the Euclidean distance between two landmarks in a 2D space.
  
- `detect_gestures(landmarks)`: Detects hand gestures based on landmarks and simulates corresponding mouse actions using PyAutoGUI.

### Main Loop

The main loop captures frames from the webcam feed, processes them to detect hand landmarks, and invokes the `detect_gestures` function to infer and act upon the detected gestures. The output frame with drawn landmarks is displayed using OpenCV.

Press `q` to exit the loop and close the application.

## Running the Code

### Rule-based approach (original)

To run the code, save the above code in a Python file (e.g., `hand_gesture_control.py`) and execute:

```bash
python hand_gesture_control.py
```

### Keras-based approach

```bash
python keras_gesture_control.py --model path/to/your/model.h5 --confidence 0.5
```

Arguments:
- `--model`: Path to your trained Keras model file (.h5 or SavedModel directory)
- `--confidence`: Minimum confidence threshold for gesture detection (0.0-1.0, default: 0.5)
- `--standby-timeout`: Seconds of inactivity before entering standby mode (default: 60)
- `--no-low-attention`: Disable low-attention/standby mode (always run at full CPU)

### Low-Attention Mode (Background Operation)

The Keras-based gesture control includes a **low-attention mode** designed for running in the background while you work. This mode significantly reduces CPU usage when you're not actively using gestures.

**How it works:**
1. **Active Mode**: Full gesture detection and processing
2. **Standby Mode**: After 60 seconds (configurable) of no detected gestures, the system enters standby
3. **Wake Up**: Touch both hands together to wake the system from standby

**Benefits:**
- Reduced CPU load when not in use (processes only every 10th frame in standby)
- Visual indicator shows current mode (STANDBY message displayed)
- Automatic transition between modes based on activity

**Example usage for background operation:**
```bash
# Run with 2-minute standby timeout
python keras_gesture_control.py --model gesture_model.h5 --standby-timeout 120

# Disable standby mode for continuous operation
python keras_gesture_control.py --model gesture_model.h5 --no-low-attention
```

Ensure you have the necessary packages installed and a webcam connected to your system. Adjust the camera index in `cv2.VideoCapture(0)` if using an external camera.

## Notes

1. Ensure proper lighting conditions for accurate hand detection.
2. Adjust the `threshold_angle` and other parameters for better gesture detection based on your environment and needs.
3. For Keras-based classification, you'll need to train the model with your own gesture dataset. The `create_sample_model()` function provides a starting architecture.

## Training Your Own Keras Model

To train your own gesture classifier:

1. Collect hand landmark data for each gesture you want to recognize
2. Use the `landmarks_to_features()` method from `GestureClassifier` to convert landmarks to features
3. Train the model using standard Keras training methods
4. Save the trained model and use it with `keras_gesture_control.py`

Example training setup:
```python
from keras_gesture_classifier import GestureClassifier, create_sample_model
import numpy as np

# Create a model
model = create_sample_model(num_gestures=6, save_path="my_model.h5")

# Prepare your training data (X: features, y: gesture labels as integers)
# X should be shape (num_samples, 63) - 21 landmarks * 3 coordinates
# y should be shape (num_samples,) - integer labels 0 to num_gestures-1

# Train the model
# model.fit(X_train, y_train, epochs=50, validation_split=0.2)
# model.save("trained_gesture_model.h5")
```

## License

This code is provided under the MIT License. Make sure you mention the original source and author if you use or modify this code.