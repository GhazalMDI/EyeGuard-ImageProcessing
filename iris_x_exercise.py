import cv2
import mediapipe as mp
from collections import deque

class iris_x_exercise:
    def __init__(self):
        # ساخت مدل فقط یکبار
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ایندکس‌ها
        self.leftIrisIndx = [474, 475, 476, 477]
        self.rightIrisIndx = [469, 470, 471, 472]

        self.leftCorner = (33, 133)
        self.rightCorner = (362, 263)

        # بافر نرم‌سازی
        SMOOTH_N = 5
        self.left_buf = deque(maxlen=SMOOTH_N)
        self.right_buf = deque(maxlen=SMOOTH_N)

        # thresholds
        self.LEFT_TH = 0.05
        self.RIGHT_TH = -0.05

        # state
        self.state = "neutral"
        self.counter = 0
        self.TARGET_REPS = 5


    def iris_center(self, landmarks, iris_idxs, w, h):
        xs = [landmarks[i].x * w for i in iris_idxs]
        ys = [landmarks[i].y * h for i in iris_idxs]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def eye_center_and_width(self, landmarks, corner_idxs, w, h):
        x1 = landmarks[corner_idxs[0]].x * w
        x2 = landmarks[corner_idxs[1]].x * w
        cx = (x1 + x2) / 2.0
        width = abs(x2 - x1) + 1e-6
        return cx, width


    def process_frame(self, frame):
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # مرکز ایریس
            left_ix, left_iy = self.iris_center(lm, self.leftIrisIndx, w, h)
            right_ix, right_iy = self.iris_center(lm, self.rightIrisIndx, w, h)

            # مرکز چشم
            left_eye_cx, left_eye_w = self.eye_center_and_width(lm, self.leftCorner, w, h)
            right_eye_cx, right_eye_w = self.eye_center_and_width(lm, self.rightCorner, w, h)

            # نسبی‌سازی
            left_rel_x = (left_ix - left_eye_cx) / left_eye_w
            right_rel_x = (right_ix - right_eye_cx) / right_eye_w

            # نرم‌سازی
            self.left_buf.append(left_rel_x)
            self.right_buf.append(right_rel_x)
            avg_left = sum(self.left_buf) / len(self.left_buf)
            avg_right = sum(self.right_buf) / len(self.right_buf)
            avg_both = (avg_left + avg_right) / 2.0


            # ---- STATE MACHINE ----
            if avg_both < self.RIGHT_TH:     # نگاه راست
                if self.state == "looking_left":
                    self.state = "looking_right"
                    self.counter += 1
                    print("REP:", self.counter)
                    if self.counter >= self.TARGET_REPS:
                        return frame, True

            elif avg_both > self.LEFT_TH:   # نگاه چپ
                if self.state == "neutral":
                    self.state = "looking_left"

            else:
                if self.state != "looking_left":
                    self.state = "neutral"

            # رسم
            cv2.circle(frame, (int(left_ix), int(left_iy)), 3, (0,255,0), -1)
            cv2.circle(frame, (int(right_ix), int(right_iy)), 3, (0,255,0), -1)

            cv2.putText(frame, f"Reps: {self.counter}/{self.TARGET_REPS}",
                        (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0),2)

        return frame, False