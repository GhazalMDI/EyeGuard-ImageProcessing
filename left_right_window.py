import cv2
from ultralytics import YOLO
from  neck_movement import found_neck,check_move_right_left


def left_right():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n-face.pt")  
    target_count = 5  
    threshold = 20        

    count = 0
    last_center_x = None
    direction = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not read")
            break

        frame = cv2.flip(frame, 1)
        results = model(frame,stream=True) 

        current_center_x = found_neck(results, frame,x=True)
        last_center_x, direction, count = check_move_right_left(
            current_center_x, last_center_x, direction, count, threshold
        )
        cv2.putText(frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Neck Movement", frame)

        if count >= target_count:
            break

        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
