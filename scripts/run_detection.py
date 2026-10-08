from pathlib import Path
import cv2
from src.detection.yolo_detector import YoloDetector

sequence_path = Path("datasets/MOT17/train/MOT17-05-FRCNN/img1")
frames = sorted(sequence_path.glob("*.jpg"))
class_names = ["person"]
print(f"Found {len(frames)} frames in the sequence.")

# Load a pretrained YOLO model (e.g., YOLOv8)
detector = YoloDetector("yolo26n.pt", class_names) # Replace with the path to your YOLO model

for frame_path in frames:
    frame = cv2.imread(str(frame_path))
    
    if frame is None:
        print(f"Failed to read frame: {frame_path}")
        continue
    print(f"Read frame: {frame_path}, shape: {frame.shape}")
    
    # Perform object detection on the frame
    detections = detector.detect(frame, confidence_threshold=0.5)
    for detection in detections:
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        confidence = detection.confidence
        class_name = detection.class_name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {class_name}, Confidence: {confidence:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(70) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()