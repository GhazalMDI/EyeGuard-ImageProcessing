import cv2
import time

from ultralytics import YOLO
from  neck_movement import found_neck,check_move_right_left


def left_right():
    cap = cv2.VideoCapture(0)
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
        out = found_neck(results, frame, x=True)
        if out is not None:
            baseline_angle = out
            last_angle = baseline_angle
            print(f"Baseline angle = {baseline_angle}")



    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"{i}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
        cv2.imshow("Neck Movement", frame)
        cv2.waitKey(1000)

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, "Start!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
        cv2.imshow("Neck Movement", frame)
        cv2.waitKey(1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = model(frame, stream=True)
        current_angle = found_neck(results, frame, x=True)

        if current_angle is None:
            cv2.putText(frame, "Face not detected!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Neck Movement", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue


        current_angle, direction, count = check_move_right_left(
            current_angle=current_angle,
            last_angle=last_angle,
            direction=direction,
            count=count,
            baseline_angle=baseline_angle
        )
        last_angle = current_angle 

        max_angle = 30
        angle_diff = abs(current_angle - baseline_angle)
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