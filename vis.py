# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 05:25:09 2024

@author: Yoush
"""

from ultralytics import YOLO
import cv2
import time

model_path = 'G:/Yolov11Swin/ultralytics-main/best.pt'
# Load a COCO-pretrained YOLO11n model
model = YOLO(model_path)


#results = model("C:/Users/Yoush/Downloads/Yolov11_Swin/ultralytics-main/test.jpg", imgsz =640)



#results[0].show()




video_path = 'G:/Yolov11Swin/ultralytics-main/video.mp4'




output_video_path = "G:/Yolov11Swin/ultralytics-main/out_video.mp4"




# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Get the frame width, height, and frames per second (FPS) from the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' or 'mp4v'
out = cv2.VideoWriter(output_video_path, fourcc, input_fps, (frame_width, frame_height))

# Initialize variables for FPS calculation
frame_count = 0
total_time = 0.0

# Optional: Font settings for overlaying text
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
color = (0, 255, 0)  # Green color for text
thickness = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # If the frame is not read correctly, stop the loop

    start_time = time.time()  # Start time for the current frame

    # Run inference on the frame (resize it if needed)
    results = model(frame)  # You can also specify imgsz=640 here if necessary

    # Since results is a list, access the first item in the list
    result = results[0]

    # Plot the detections on the frame (this modifies the frame with bounding boxes)
    frame_with_detections = result.plot()  # Plot returns the frame with detections

    end_time = time.time()  # End time for the current frame
    processing_time = end_time - start_time
    total_time += processing_time
    frame_count += 1

    # Calculate average FPS
    if frame_count > 0:
        average_fps = frame_count / total_time
    else:
        average_fps = 0.0

    # Optional: Overlay average FPS on the frame
    fps_text = f'Avg FPS: {average_fps:.2f}'
    cv2.putText(frame_with_detections, fps_text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

    # Write the frame with detections to the output video
    out.write(frame_with_detections)

    # Optionally, display the frame with detections
    cv2.imshow("Detected Frame", frame_with_detections)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After processing all frames, calculate final average FPS
if total_time > 0:
    final_average_fps = frame_count / total_time
else:
    final_average_fps = 0.0

print(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
print(f"Average FPS: {final_average_fps:.2f}")

# Release the video capture and writer objects
cap.release()
out.release()

# Close the display window
cv2.destroyAllWindows()





# # Open the video using OpenCV
# cap = cv2.VideoCapture(video_path)

# # Get the frame width, height, and frames per second (FPS) from the video
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)

# # Define the codec and create VideoWriter to save the output video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID' or 'mp4v'
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break  # If the frame is not read correctly, stop the loop

#     # Run inference on the frame (resize it if needed)
#     results = model(frame)  # You can also specify imgsz=640 here if necessary

#     # Since results is a list, access the first item in the list
#     result = results[0]

#     # Plot the detections on the frame (this modifies the frame with bounding boxes)
#     frame_with_detections = result.plot()  # Plot returns the frame with detections

#     # Write the frame with detections to the output video
#     out.write(frame_with_detections)

#     # Optionally, display the frame with detections
#     cv2.imshow("Detected Frame", frame_with_detections)

#     # Exit loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and writer objects
# cap.release()
# out.release()

# # Close the display window
# cv2.destroyAllWindows()








from ultralytics import YOLO
import cv2
import os
import time

# Paths
model_path = 'G:/Yolov11Swin/ultralytics-main/bestt.pt'
input_folder = 'G:/Yolov11Swin/ultralytics-main/Images'  # Folder containing input images
output_folder = 'G:/Yolov11Swin/ultralytics-main/yolov11swin_img'  # Folder to save output images

# Load the model
model = YOLO(model_path)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all image files in the input folder (common image extensions)
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# Initialize FPS tracking
frame_count = 0
total_time = 0.0

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    output_image_path = os.path.join(output_folder, f"swin_{image_name}")

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_name}, skipping.")
        continue

    start_time = time.time()

    # Run YOLO inference
    results = model(image)
    result = results[0]

    # Plot detections
    image_with_detections = result.plot()

    end_time = time.time()
    processing_time = end_time - start_time
    total_time += processing_time
    frame_count += 1
 
    # Optional: Overlay average FPS so far
    avg_fps = frame_count / total_time if total_time > 0 else 0
    fps_text = f'Avg FPS: {avg_fps:.2f}'
    cv2.putText(image_with_detections, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the result
    cv2.imwrite(output_image_path, image_with_detections)
    print(f"Processed {image_name} -> Saved to {output_image_path}")

    # Optional: Display the result
    cv2.imshow("Detection", image_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final FPS summary
print(f"Processed {frame_count} images in {total_time:.2f} seconds.")
if total_time > 0:
    print(f"Average FPS: {frame_count / total_time:.2f}")
else:
    print("Average FPS: N/A (no images processed)")

cv2.destroyAllWindows()















from ultralytics import YOLO
import cv2
import os
import time

# Paths
model_path = 'G:/Yolov11Swin/ultralytics-main/best_swin.pt'
input_folder = 'G:/Yolov11Swin/ultralytics-main/test/0f0d7759-fa6e-3296-b528-6c862d061bdd/ring_front_center'  # Folder containing input images
output_folder = 'G:/Yolov11Swin/ultralytics-main/yoloswin_images'  # Folder to save output images

# Load the model
model = YOLO(model_path)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    output_image_path = os.path.join(output_folder, f"swin_{image_name}")

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_name}, skipping.")
        continue

    # Run YOLO inference
    results = model(image)
    result = results[0]

    # Plot detections with smaller label font and thinner boxes
    image_with_detections = result.plot(line_width=1, font_size=0.3)

    # Save the result
    cv2.imwrite(output_image_path, image_with_detections)
    print(f"Processed {image_name} -> Saved to {output_image_path}")

    # Optional: Display the result
    cv2.imshow("Detection", image_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()







from ultralytics import YOLO
import cv2
import os
import time

# Paths
model_path = 'G:/Yolov11Swin/ultralytics-main/best_v11.pt'
input_folder = 'G:/Yolov11Swin/ultralytics-main/test/0f0d7759-fa6e-3296-b528-6c862d061bdd/ring_front_center'  # Folder containing input images
output_folder = 'G:/Yolov11Swin/ultralytics-main/yolov11_images'  # Folder to save output images

# Load the model
model = YOLO(model_path)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    output_image_path = os.path.join(output_folder, f"v11_{image_name}")

    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_name}, skipping.")
        continue

    # Run YOLO inference
    results = model(image)
    result = results[0]

    # Plot detections with smaller label font and thinner boxes
    image_with_detections = result.plot(line_width=1, font_size=0.3)

    # Save the result
    cv2.imwrite(output_image_path, image_with_detections)
    print(f"Processed {image_name} -> Saved to {output_image_path}")

    # Optional: Display the result
    cv2.imshow("Detection", image_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

































































