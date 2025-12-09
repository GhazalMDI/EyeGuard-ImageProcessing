import cv2
import time
from ultralytics import YOLO
from neck_movement import found_neck

class left_right:
    def __init__(self, target_count=5):
        self.model = YOLO("yolov8n-face.pt")  
        self.target_count = target_count
        self.count = 0
        self.direction = None
        self.baseline_angle = None
        self.last_angle = None
        time.sleep(1)

    def process_frame(self, frame):
        # تعیین baseline
        while self.baseline_angle is None:
            results = self.model(frame, stream=True)
            out = found_neck(results, frame, x=True)
            if out is not None:
                self.baseline_angle = out
                self.last_angle = self.baseline_angle
                print(f"Baseline angle = {self.baseline_angle}")

        # تشخیص زاویه فعلی
        results = self.model(frame, stream=True)
        current_angle = found_neck(results, frame, x=True)

        if current_angle is None:
            # بجای نوشتن روی فریم، فقط return count
            return frame, False, self.count

        # بررسی حرکت چپ/راست
        current_angle, self.direction, self.count = self.check_move_right_left(
            current_angle=current_angle,
            last_angle=self.last_angle,
            direction=self.direction,
            count=self.count,
            baseline_angle=self.baseline_angle
        )
        self.last_angle = current_angle 

        # چک کردن تکمیل تمرین
        if self.count >= self.target_count:
            print("Exercise complete!")
            return frame, True, self.count

        # بازگشت فریم بدون نوشتن روی آن، و count جدا
        return frame, False, self.count
    
    @staticmethod
    def check_move_right_left(current_angle, last_angle, direction, count, baseline_angle):
        if current_angle is None or last_angle is None or baseline_angle is None:
            return current_angle, direction, count

        angle_diff = current_angle - baseline_angle

        # وقتی سر به سمت راست حرکت کرد
        if -3 <= angle_diff <= 3 and direction != "right":
            direction = "right"
            count += 1
        elif angle_diff < -3 or angle_diff > 3:
            direction = None 

        return current_angle, direction, count
