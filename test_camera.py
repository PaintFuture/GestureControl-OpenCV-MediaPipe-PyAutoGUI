import cv2

print("Testing camera access...")
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("ERROR: Could not open camera!")
else:
    print("Camera opened successfully!")
    ret, frame = capture.read()
    if ret:
        print(f"Frame captured successfully! Shape: {frame.shape}")
        cv2.imshow("Test Camera", frame)
        print("Press any key in the window to close...")
        cv2.waitKey(0)
    else:
        print("ERROR: Could not read frame from camera!")

capture.release()
cv2.destroyAllWindows()
print("Test completed.")
