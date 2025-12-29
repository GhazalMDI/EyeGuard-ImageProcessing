import cv2
import mediapipe as mp
import math
import time

class find_movement:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.state = "CALIB_OPEN"
        self.state_start_time = time.time()
        self.open_samples = {"Left": [], "Right": []}
        self.close_samples = {"Left": [], "Right": []}
        self.open_mean = {"Left": None, "Right": None}
        self.close_mean = {"Left": None, "Right": None}
        self.threshold = {"Left": None, "Right": None}
        self.initial_count = 10
        self.target_count = 10
        self.register_cooldown = 0.35
        self.last_register_time = 0.0

        # برای جلوگیری از شمارشِ فریم اول
        self.last_registered_status = None

        # برای کاهش نویز
        self.stable_status = None
        self.stable_frames = 0
        self.required_stable_frames = 3

    @staticmethod
    def _dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def avg_finger_distance(self, hand_landmarks):
        """
        میانگین فاصله نوک انگشت‌ها تا مچ (نرمال شده)
        """
        lm = [(p.x, p.y) for p in hand_landmarks.landmark]
        wrist = lm[0]
        finger_tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
        return sum(self._dist(wrist, tip) for tip in finger_tips) / 5.0

    @staticmethod
    def _hand_label(results, idx):
        """
        برچسب Left / Right از mediapipe
        """
        try:
            return results.multi_handedness[idx].classification[0].label  # "Left" / "Right"
        except Exception:
            return None

    def _get_two_hands_dist(self, results):
        """
        اگر دقیقاً دو دست داشتیم، یک دیکشنری برمی‌گرداند:
        {"Left": dist, "Right": dist}
        اگر دست‌ها ناقص بودند None برمی‌گرداند
        """
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) != 2:
            return None

        d = {}
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            label = self._hand_label(results, i)
            if label not in ("Left", "Right"):
                return None
            d[label] = self.avg_finger_distance(hand_lm)

        # باید هر دو دست موجود باشد
        if "Left" not in d or "Right" not in d:
            return None
        return d

    @staticmethod
    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if self.state == "CALIB_OPEN":
            d = self._get_two_hands_dist(results)
            if d:
                self.open_samples["Left"].append(d["Left"])
                self.open_samples["Right"].append(d["Right"])


            if time.time() - self.state_start_time >= 1.5:
                left_mean = self._mean(self.open_samples["Left"])
                right_mean = self._mean(self.open_samples["Right"])
                if left_mean is not None and right_mean is not None:
                    self.open_mean["Left"] = left_mean
                    self.open_mean["Right"] = right_mean
                    self.state = "CALIB_CLOSE"
                    self.state_start_time = time.time()
                else:
                    self.state_start_time = time.time()

            return frame, False, self.initial_count - self.target_count

        if self.state == "CALIB_CLOSE":
            d = self._get_two_hands_dist(results)
            if d:
                self.close_samples["Left"].append(d["Left"])
                self.close_samples["Right"].append(d["Right"])

            if time.time() - self.state_start_time >= 1.5:
                left_mean = self._mean(self.close_samples["Left"])
                right_mean = self._mean(self.close_samples["Right"])
                if left_mean is not None and right_mean is not None:
                    self.close_mean["Left"] = left_mean
                    self.close_mean["Right"] = right_mean


                    self.threshold["Left"] = (self.open_mean["Left"] + self.close_mean["Left"]) / 2.0
                    self.threshold["Right"] = (self.open_mean["Right"] + self.close_mean["Right"]) / 2.0


                    self.state = "RUNNING"
                    self.last_registered_status = None
                    self.stable_status = None
                    self.stable_frames = 0
                    self.last_register_time = time.time()
                else:
                    self.state_start_time = time.time()

            return frame, False, self.initial_count - self.target_count


        if self.state == "RUNNING":
            d = self._get_two_hands_dist(results)
            combined_status = None

            if d:
                # تعیین وضعیت هر دست
                left_status = "close" if d["Left"] < self.threshold["Left"] else "open"
                right_status = "close" if d["Right"] < self.threshold["Right"] else "open"

                if left_status == right_status:
                    combined_status = left_status

            if combined_status is None:
                self.stable_status = None
                self.stable_frames = 0
            else:
                if combined_status == self.stable_status:
                    self.stable_frames += 1
                else:
                    self.stable_status = combined_status
                    self.stable_frames = 1


            if self.stable_status and self.stable_frames >= self.required_stable_frames:
                now = time.time()

                if self.last_registered_status is None:
                    self.last_registered_status = self.stable_status
                    self.last_register_time = now
                else:

                    if (
                            self.stable_status != self.last_registered_status
                            and (now - self.last_register_time) >= self.register_cooldown
                    ):
                        self.target_count -= 1
                        self.last_registered_status = self.stable_status
                        self.last_register_time = now

            done = self.target_count <= 0
            return frame, done, self.initial_count - self.target_count

        return frame, False, self.initial_count - self.target_count