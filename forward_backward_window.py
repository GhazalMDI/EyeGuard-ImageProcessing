import cv2
from ultralytics import YOLO
from  neck_movement import found_neck,check_forward_backward


def forward_backward():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n-face.pt")  
    target_count = 5  
    threshold = 20       
    count = 0
    last_center_y = None
    direction = None



    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not read")
            break

        frame = cv2.flip(frame, 1)
        results = model(frame,stream=True) 

        current_center_y = found_neck(results, frame,y=True)
        last_center_y, direction, count = check_forward_backward(
            current_center_y, last_center_y, direction, count, threshold
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
