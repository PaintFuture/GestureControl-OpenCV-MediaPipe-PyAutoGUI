import cv2
import sys

print("=== Camera Diagnostics ===")
print(f"OpenCV Version: {cv2.__version__}")
print("\nAttempting to open camera (index 0)...")

# Try to open camera with timeout
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow for Windows

if not capture.isOpened():
    print("❌ FAILED: Could not open camera with index 0")
    print("\nPossible issues:")
    print("1. No camera connected")
    print("2. Camera in use by another application")
    print("3. Camera permissions not granted")
    print("4. Driver issues")
    sys.exit(1)

print("✓ Camera opened successfully!")

# Get camera properties
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

print(f"\nCamera Properties:")
print(f"  Resolution: {int(width)}x{int(height)}")
print(f"  FPS: {fps}")

# Try to read a frame
print("\nReading test frame...")
ret, frame = capture.read()

if not ret:
    print("❌ FAILED: Could not read frame from camera")
    capture.release()
    sys.exit(1)

print(f"✓ Frame read successfully! Shape: {frame.shape}")
print("\nDisplaying test window...")
print("Press any key in the window to close")

cv2.imshow("Camera Test - Press any key to close", frame)
cv2.waitKey(0)

capture.release()
cv2.destroyAllWindows()

print("\n✓ All tests passed! Camera is working correctly.")
