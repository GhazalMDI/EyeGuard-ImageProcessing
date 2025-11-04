import cv2
import mediapipe as mp
import math

class PoseTracker:
    def __init__(self, angle_threshold=15):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.angle_threshold = angle_threshold  
        self.last_direction = None
        self.reps = 0

    def calculate_angle(self, a, b, c):
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        ang = abs(ang)
        if ang > 180:
            ang = 360 - ang
        return ang

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            shoulder_left = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            shoulder_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            nose = [landmarks[self.mp_pose.PoseLandmark.NOSE].x,
                    landmarks[self.mp_pose.PoseLandmark.NOSE].y]

            angle = self.calculate_angle(shoulder_left, nose, shoulder_right)

            direction = None
            if angle > 90 + self.angle_threshold:
                direction = "Left"
            elif angle < 90 - self.angle_threshold:
                direction = "Right"

            if direction and direction != self.last_direction:
                self.reps += 1
                self.last_direction = direction

            return angle, direction, self.reps

        return None, None, self.reps
