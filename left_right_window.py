import cv2
import time

from ultralytics import YOLO
from  neck_movement import found_neck,check_move_right_left


class left_right:
    def __init__(self):
        
        self.model = YOLO("yolov8n-face.pt")  

        self.target_count = 5
        self.count = 0
        self.direction = None
        self.baseline_angle = None
        self.last_angle = None

    print("CALIBRATION → لطفاً صاف بنشینید...")
    time.sleep(1)
    def process_frame(self,frame):
        while self.baseline_angle is None:
            results = self.model(frame, stream=True)
            out = found_neck(results, frame, x=True)
            if out is not None:
                self.baseline_angle = out
                self.last_angle = self.baseline_angle
                print(f"Baseline angle = {self.baseline_angle}")
   
        results = self.model(frame, stream=True)
        current_angle = found_neck(results, frame, x=True)

        if current_angle is None:
            cv2.putText(frame, "Face not detected!", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        


        current_angle, self.direction, self.count = check_move_right_left(
            current_angle=current_angle,
            last_angle=self.last_angle,
            direction=self.direction,
            count=self.count,
            baseline_angle=self.baseline_angle
        )
        self.last_angle = current_angle 

        max_angle = 30
        angle_diff = abs(current_angle - self.baseline_angle)
        progress = min(angle_diff / max_angle, 1.0)
        bar_w = int(progress * 300)

        cv2.putText(frame, f"Count: {self.count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        if self.count >= self.target_count:
            print("Exercise complete!")
            return True
        return frame,False

        