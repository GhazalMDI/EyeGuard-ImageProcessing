import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("yolov8n-face.pt")  
target_count = 5 
threshold = 60        

count = 0
last_center_x = None
direction = None

def found_neck(results, frame,x=None,y=None):
    current_center_x = None
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            if label == "face":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if x:
        current_center_x = (x1 + x2) // 2
        return current_center_x
    else:
        current_center_y = (y1 + y2) // 2
        return current_center_y

def check_move_right_left(current_center_x, last_center_x, direction, count, threshold):
    if last_center_x is not None and current_center_x is not None:
        movement = current_center_x - last_center_x
        if movement > threshold and direction != "right":
            count += 1
            direction = "right"
            last_center_x = current_center_x
        elif movement < -threshold and direction != "left":
            count += 1
            direction = "left"
            last_center_x = current_center_x
    elif current_center_x is not None:
        last_center_x = current_center_x

    return last_center_x, direction, count


def check_forward_backward(current_center_y,last_center_y,direction,count,threshold):
    if last_center_y is not None and current_center_y is not None:
        movement_y = current_center_y -  last_center_y 
        if movement_y > threshold and direction !="forward":
            count+=1
            direction = "forward"
            last_center_y = current_center_y
        elif movement_y < -threshold  and direction !="backward":
            count+=1
            direction = "backward"
            last_center_y = current_center_y
    elif current_center_y is not None:
        last_center_y = current_center_y
    return last_center_y,direction,count

