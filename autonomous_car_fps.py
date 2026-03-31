import os
import time
import cv2
from ultralytics import YOLO

# Define the model path
model_path = "yolo-swin.pt"
video_path = "Cars_Moving.mp4"
# Initialize video capture
cap = cv2.VideoCapture(video_path)  # Replace with your video path
ret, frame = cap.read()
if not ret:
    print("Error: Unable to read video file. Please check the path.")
    exit()

# Load the YOLO model
model = YOLO(model_path)

# FPS calculation variables
fps_list = []

while ret:
    frame_start_time = time.time()

    # Perform YOLO detection
    results = model(frame)

    # Process the results (optional, can be removed if not needed)
    # ... (your processing code here)

    # Calculate FPS
    latency = (time.time() - frame_start_time) * 1000  # milliseconds
    fps = 1000 / latency
    fps_list.append(fps)

    # Display the frame (optional, can be commented out)
    # cv2.imshow("Frame", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

# Calculate and print average FPS
if fps_list:
    average_fps = sum(fps_list) / len(fps_list)
    print(f"Average FPS: {average_fps:.2f}")
else:
    print("Average FPS: N/A (no frames were processed)")