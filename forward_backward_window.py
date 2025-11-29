import cv2
import time
import math

from ultralytics import YOLO
from  neck_movement import found_neck,check_forward_backward



def get_face_angle(results):
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            return None
        
        box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dx, dy))

        return angle, cy

    return None


def forward_backward():
    cap = cv2.VideoCapture(0)

    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"{i}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
        cv2.imshow("Neck Movement", frame)
        cv2.waitKey(1000) 

    
    model = YOLO("yolov8n-face.pt")

    target_count = 5
    count = 0
    direction = None
    baseline_angle = None
    last_angle = None

    print("CALIBRATION → لطفاً صاف بنشینید...")
    time.sleep(1)

    while baseline_angle is None:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        results = model(frame, stream=True)
        out = get_face_angle(results)
        if out:
            baseline_angle, _ = out
            last_angle = baseline_angle
            print(f"Baseline angle = {baseline_angle}")


    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = model(frame, stream=True)
        out = get_face_angle(results)

        if out is None:
            cv2.putText(frame, "Face not detected!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Neck Movement", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        angle, cy = out
        angle_diff = abs(angle - baseline_angle)

        if time.time() - start_time >= 3:
            angle, direction, count = check_forward_backward(
                current_angle=angle,
                last_angle=last_angle,
                direction=direction,
                count=count,
                baseline_angle=baseline_angle
            )

        last_angle = angle 

        max_angle = 30
        progress = min(angle_diff / max_angle, 1.0)
        bar_w = int(progress * 300)
        cv2.rectangle(frame, (20, 80), (20 + bar_w, 110), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 80), (320, 110), (255, 255, 255), 2)

        cv2.putText(frame, f"Angle: {int(angle_diff)} deg", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Neck Movement", frame)

        if count >= target_count:
            print("Exercise complete!")
            break

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
