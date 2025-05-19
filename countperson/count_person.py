import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Start webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change to video path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame)

    # Count the number of persons detected
    person_count = sum(1 for obj in results[0].boxes.cls if results[0].names[int(obj)] == "person")

    # Get processed frame with bounding boxes
    result_frame = results[0].plot()

    # Display the count on the frame
    cv2.putText(result_frame, f"Persons: {person_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("People Counting - YOLOv8", result_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
