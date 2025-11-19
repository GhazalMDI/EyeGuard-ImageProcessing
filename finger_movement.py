import cv2
import mediapipe as mp
import math
import time

def find_movement():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def avg_finger_distance(hand_landmarks):
        landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        wrist = landmarks[0]
        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        return sum([distance(wrist, tip) for tip in finger_tips]) / 5.0

    def draw_status(frame, text, y=50):
        cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    def calibrate(cap, message, wait_time=2):
        print(message)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) < 2:
                    draw_status(frame, "Show both hands to camera", y=50)
                    cv2.imshow("Calibration", frame)
                    cv2.waitKey(1)
                    continue
                avg_dists = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    avg_dist = avg_finger_distance(hand_landmarks)
                    avg_dists.append(avg_dist)
                    draw_status(frame, f"Hand {i+1} Distance: {avg_dist:.3f}", y=50 + i*50)
                draw_status(frame, message, y=50 + len(avg_dists)*50)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1)
                time.sleep(wait_time)
                return avg_dists
            else:
                draw_status(frame, "Show both hands to camera", y=50)
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1)

    cap = cv2.VideoCapture(0)
    open_dists = calibrate(cap, "Calibration: Show your OPEN hands")
    close_dists = calibrate(cap, "Calibration: Show your CLOSED hands")
    thresholds = [(o + c) / 2 for o, c in zip(open_dists, close_dists)]

    target_count = 10
    last_registered_status = None  
    register_cooldown = 0.5 
    last_register_time = 0

    while target_count > 0:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        status_list = []

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                avg_dist = avg_finger_distance(hand_landmarks)
                threshold = thresholds[i]
                status = "close" if avg_dist < threshold else "open"
                status_list.append(status)
                draw_status(frame, f"Hand {i+1}: {status}", y=50 + i*50)

            if (status_list[0] == status_list[1] and 
                status_list[0] != last_registered_status and  
                time.time() - last_register_time > register_cooldown):  
                target_count -= 1
                last_registered_status = status_list[0]
                last_register_time = time.time()
                draw_status(frame, f"Registered! Remaining: {target_count}", y=150)

        else:
            draw_status(frame, "Show BOTH hands to camera", y=50)

        draw_status(frame, f"Remaining repetitions: {target_count}", y=200)
        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exercise completed!")
