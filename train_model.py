"""
Train the Keras gesture classification model.

This script trains the model using collected gesture data.
"""

import json
import numpy as np
import sys
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras_gesture_classifier import create_sample_model


def load_training_data(filepath):
    """Load training data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def prepare_dataset(training_data):
    """Convert training data to numpy arrays for training."""
    
    X = []  # Features
    y = []  # Labels
    
    gesture_labels = ["none", "pinch", "left_click", "right_click", "scroll_up", "scroll_down"]
    
    for gesture_idx, gesture_name in enumerate(gesture_labels):
        samples = training_data.get(gesture_name, [])
        
        for sample in samples:
            # Normalize relative to wrist (first 3 values: x, y, z of wrist)
            sample_array = np.array(sample, dtype=np.float32)
            wrist_x, wrist_y, wrist_z = sample_array[0], sample_array[1], sample_array[2]
            
            for i in range(0, len(sample_array), 3):
                sample_array[i] -= wrist_x
                sample_array[i + 1] -= wrist_y
                sample_array[i + 2] -= wrist_z
            
            X.append(sample_array)
            y.append(gesture_idx)
    
    return np.array(X), np.array(y)


def train_model(data_filepath, output_model_path="trained_gesture_model.h5"):
    """Train the gesture classification model."""
    
    print("Loading training data...")
    training_data = load_training_data(data_filepath)
    
    # Check data
    gesture_labels = ["none", "pinch", "left_click", "right_click", "scroll_up", "scroll_down"]
    print("\nSamples per gesture:")
    for gesture in gesture_labels:
        count = len(training_data.get(gesture, []))
        print(f"  {gesture}: {count}")
    
    total_samples = sum(len(samples) for samples in training_data.values())
    if total_samples < 30:
        print(f"\n⚠ WARNING: Only {total_samples} total samples. Recommended: 300+ (50+ per gesture)")
        print("The model may not work well with so little data.")
    
    print("\nPreparing dataset...")
    X, y = prepare_dataset(training_data)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create model
    print("\nCreating model...")
    model = keras.Sequential([
        keras.layers.Input(shape=(63,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining model...")
    print("(This may take a few minutes...)\n")
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate
    print("\nFinal Results:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Training accuracy: {train_acc*100:.2f}%")
    print(f"Validation accuracy: {val_acc*100:.2f}%")
    
    # Save model
    model.save(output_model_path)
    print(f"\n✓ Model saved to {output_model_path}")
    
    print("\nNext step:")
    print(f"  Run: python keras_gesture_control.py --model {output_model_path}")
    
    return model, history


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_model.py <training_data_file.json> [output_model.h5]")
        print("\nExample:")
        print("  python train_model.py training_data/gesture_data_20251130_120000.json")
        sys.exit(1)
    
    data_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "trained_gesture_model.h5"
    
    train_model(data_file, output_file)
