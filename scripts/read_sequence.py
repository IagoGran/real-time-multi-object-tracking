from pathlib import Path

import cv2

sequence_path = Path("datasets/MOT17/train/MOT17-05-FRCNN/img1")

frames = sorted(sequence_path.glob("*.jpg"))

print(f"Found {len(frames)} frames in the sequence.")

for frame_path in frames:
    frame = cv2.imread(str(frame_path))
    
    if frame is None:
        print(f"Failed to read frame: {frame_path}")
        continue
    print(f"Read frame: {frame_path}, shape: {frame.shape}")
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(70) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()