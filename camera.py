import cv2
import time

# Open your Mac's built-in camera (0 = default)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not access the camera.")
    exit()

print("📷 Starting camera... capturing in 1 second.")
time.sleep(1)  # wait for camera to adjust lighting

ret, frame = camera.read()
if not ret:
    print("❌ Failed to capture image.")
else:
    filename = "capture.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ Image saved as {filename}")

camera.release()
cv2.destroyAllWindows()