import cv2
import mediapipe as mp

class blink:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # نقاط پلک‌ها
        self.LEFT_UP = 159
        self.LEFT_DOWN = 145
        self.RIGHT_UP = 386
        self.RIGHT_DOWN = 374

        self.THRESH = 0.018

        self.blink_count = 0
        self.state = "open"   # open → closed → open → blink detected


    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            l_dist = abs(lm[self.LEFT_UP].y - lm[self.LEFT_DOWN].y)
            r_dist = abs(lm[self.RIGHT_UP].y - lm[self.RIGHT_DOWN].y)
            eye_dist = (l_dist + r_dist) / 2.0

            # ---- BLink State Machine ----
            if eye_dist < self.THRESH:   # چشم بسته
                if self.state == "open":
                    self.state = "closed"

            else:                        # چشم باز شد = پایان چشمک
                if self.state == "closed":
                    self.blink_count += 1
                    self.state = "open"

            # نمایش شمارنده
            cv2.putText(frame, f"BLINKS: {self.blink_count}/10", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,150,255), 2)

            # پایان تمرین
            if self.blink_count >= 10:
                cv2.putText(frame, "EXERCISE COMPLETE!", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 3)
                return frame, True

        return frame, False