import cv2
import math
from ultralytics import YOLO
from neck_movement import check_forward_backward


class forward_back:
    def __init__(self):
        self.model = YOLO("yolov8n-face.pt")

        self.target_count = 5
        self.count = 0
        self.direction = None

        self.baseline_angle = None     # اولین زاویه
        self.last_angle = None         # آخرین زاویه


   

    def process_frame(self, frame):


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


        # YOLO detection
        results = self.model(frame, stream=True)
        out = get_face_angle(results)

        if out is None:
            cv2.putText(frame, "Face not detected!", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return frame, False

        angle, cy = out

        # -------- baseline فقط یکبار مقدار می‌گیرد --------
        if self.baseline_angle is None:
            self.baseline_angle = angle
            self.last_angle = angle
            return frame, False

        angle_diff = abs(angle - self.baseline_angle)

        # -------- شمارش بر اساس منطق قبلی --------
        angle, self.direction, self.count = check_forward_backward(
            current_angle=angle,
            last_angle=self.last_angle,
            direction=self.direction,
            count=self.count,
            baseline_angle=self.baseline_angle
        )

        self.last_angle = angle  # ذخیره زاویه قبلی

        # -------- UI --------
        max_angle = 30
        progress = min(angle_diff / max_angle, 1.0)
        bar_w = int(progress * 300)

        cv2.putText(frame, f"Count: {self.count}/{self.target_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # -------- پایان تمرین --------
        if self.count >= self.target_count:
            return frame, True

        return frame, False