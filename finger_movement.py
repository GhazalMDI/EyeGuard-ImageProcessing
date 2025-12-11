import cv2
import mediapipe as mp
import math
import time

class find_movement:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.state = "CALIB_OPEN"

        self.open_dists = None
        self.close_dists = None
        self.thresholds = None

        self.target_count = 10
        self.initial_count = 10
        self.last_registered_status = None
        self.register_cooldown = 0.5
        self.last_register_time = 0

        self.state_start_time = time.time()

    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def avg_finger_distance(self, hand_landmarks):
        landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        wrist = landmarks[0]
        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        return sum([self.distance(wrist, tip) for tip in finger_tips]) / 5.0

    # ---------- حذف draw_status ----------
    # def draw_status(self, frame, text, y=80):
    #     cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if self.state == "CALIB_OPEN":
            # Calibration بدون متن
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                dists = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    dists.append(self.avg_finger_distance(hand_landmarks))
                self.open_dists = dists
                if time.time() - self.state_start_time > 1.5:
                    self.state = "CALIB_CLOSE"
                    self.state_start_time = time.time()
            return frame, False, self.initial_count - self.target_count

        if self.state == "CALIB_CLOSE":
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                dists = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    dists.append(self.avg_finger_distance(hand_landmarks))
                self.close_dists = dists
                if time.time() - self.state_start_time > 1.5:
                    self.thresholds = [(o + c) / 2 for o, c in zip(self.open_dists, self.close_dists)]
                    self.state = "RUNNING"
            return frame, False, self.initial_count - self.target_count

        if self.state == "RUNNING":
            status_list = []
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    avg_dist = self.avg_finger_distance(hand_landmarks)
                    threshold = self.thresholds[i]
                    status = "close" if avg_dist < threshold else "open"
                    status_list.append(status)
                if (
                    status_list[0] == status_list[1] and
                    status_list[0] != self.last_registered_status and
                    time.time() - self.last_register_time > self.register_cooldown
                ):
                    self.target_count -= 1
                    self.last_registered_status = status_list[0]

            done = self.target_count <= 0
            return frame, done, self.initial_count - self.target_count
