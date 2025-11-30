"""
Keras-based gesture classifier module.

This module provides a GestureClassifier class that uses a Keras model
for classifying hand gestures based on MediaPipe hand landmarks.
"""

import numpy as np
import os


class GestureClassifier:
    """
    A gesture classifier that uses a Keras model for prediction.
    
    The classifier expects hand landmarks from MediaPipe and converts them
    into a format suitable for the Keras model.
    """
    
    # Default gesture labels - can be overridden when loading a model
    DEFAULT_GESTURE_LABELS = [
        "none",
        "pinch",
        "left_click",
        "right_click",
        "scroll_up",
        "scroll_down",
    ]
    
    def __init__(self, model_path=None, gesture_labels=None):
        """
        Initialize the gesture classifier.
        
        Args:
            model_path: Path to a saved Keras model file (.h5 or SavedModel format).
                       If None, the classifier will use a simple default model.
            gesture_labels: List of gesture label strings corresponding to model outputs.
                           If None, uses DEFAULT_GESTURE_LABELS.
        """
        self.model = None
        self.gesture_labels = gesture_labels or self.DEFAULT_GESTURE_LABELS
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path):
        """
        Load a Keras model from the specified path.
        
        Args:
            model_path: Path to the Keras model file.
        """
        try:
            # Import tensorflow/keras here to allow the module to be imported
            # even if tensorflow is not installed
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
        except ImportError:
            raise ImportError(
                "TensorFlow/Keras is required for model-based classification. "
                "Install it with: pip install tensorflow"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    
    def landmarks_to_features(self, landmarks):
        """
        Convert MediaPipe hand landmarks to a feature vector for the model.
        
        This method normalizes the landmarks relative to the wrist position
        and flattens them into a 1D array suitable for classification.
        
        Args:
            landmarks: MediaPipe hand landmarks object containing 21 landmarks.
        
        Returns:
            numpy array of shape (1, 63) containing normalized x, y, z coordinates.
        """
        # Extract landmark coordinates
        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        
        coords = np.array(coords, dtype=np.float32)
        
        # Normalize relative to wrist (landmark 0)
        wrist_x, wrist_y, wrist_z = coords[0], coords[1], coords[2]
        for i in range(0, len(coords), 3):
            coords[i] -= wrist_x
            coords[i + 1] -= wrist_y
            coords[i + 2] -= wrist_z
        
        # Reshape for model input (batch_size=1, features=63)
        return coords.reshape(1, -1)
    
    def predict(self, landmarks, confidence_threshold=0.5):
        """
        Predict the gesture from hand landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks object.
            confidence_threshold: Minimum confidence required to return a gesture.
        
        Returns:
            Tuple of (gesture_name, confidence) if confidence >= threshold,
            otherwise returns ("none", confidence).
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Please provide a model_path when initializing "
                "the GestureClassifier or call load_model() first."
            )
        
        features = self.landmarks_to_features(landmarks)
        # Use direct model call for better performance on single predictions
        predictions = self.model(features, training=False)
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        if confidence >= confidence_threshold:
            gesture_name = self.gesture_labels[predicted_class]
        else:
            gesture_name = "none"
        
        return gesture_name, confidence
    
    def load_model(self, model_path, gesture_labels=None):
        """
        Load a new model, optionally updating gesture labels.
        
        Args:
            model_path: Path to the Keras model file.
            gesture_labels: Optional new list of gesture labels.
        """
        self._load_model(model_path)
        if gesture_labels:
            self.gesture_labels = gesture_labels


def create_sample_model(num_gestures=6, save_path="gesture_model.h5"):
    """
    Create and save a sample Keras model for gesture classification.
    
    This function creates a simple feedforward neural network that can be used
    as a starting point for training your own gesture classifier.
    
    Args:
        num_gestures: Number of gesture classes to classify.
        save_path: Path where the model will be saved.
    
    Returns:
        The created Keras model.
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError(
            "TensorFlow/Keras is required. Install it with: pip install tensorflow"
        )
    
    # Input: 21 landmarks * 3 coordinates (x, y, z) = 63 features
    model = keras.Sequential([
        layers.Input(shape=(63,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_gestures, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.save(save_path)
    print(f"Sample model saved to {save_path}")
    
    return model
