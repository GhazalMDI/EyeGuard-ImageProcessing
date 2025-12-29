# import cv2
# import time
# from ultralytics import YOLO
# model = YOLO("yolov8n-face.pt")
# target_count = 6
# count = 0
# last_center_x = None
# direction = None
# last_time_global = time.time()

def found_neck(results, frame, x=False, y=False):
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            return None
        box = r.boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        if x:
            return (x1 + x2) // 2
        if y:
            return (y1 + y2) // 2
    return None

# def check_move_right_left(current_angle, last_angle, direction, count, baseline_angle):

#     if current_angle is None or last_angle is None or baseline_angle is None:
#         return current_angle, direction, count

#     angle_diff = current_angle - baseline_angle

#     if -3 <= angle_diff <= 3 and direction != "right":
#         direction = "right"
#         count += 1
#     elif angle_diff < -3 or angle_diff > 3:
#         direction = None 
#     return current_angle, direction, count


def check_forward_backward(current_angle, last_angle, direction, count, baseline_angle):
    if current_angle is None or baseline_angle is None:
        return current_angle, direction, count

    diff = abs(current_angle - baseline_angle)
    TH = 6       # خروج از مرکز (فاصله لازم برای اینکه بگیم حرکت کرده)
    CENTER = 3   # برگشت به مرکز

    if direction is None:
        direction = "center"

    # از مرکز خارج شد
    if direction == "center" and diff > TH:
        direction = "out"

    # برگشت به مرکز => یک تکرار
    elif direction == "out" and diff < CENTER:
        count += 1
        direction = "center"

    return current_angle, direction, count