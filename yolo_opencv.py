import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv5 inference
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Display the frame with YOLOv5 annotations
    cv2.imshow('YOLOv5 with OpenCV', annotated_frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

