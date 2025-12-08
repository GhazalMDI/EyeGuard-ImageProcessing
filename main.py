from forward_backward_window import forward_back
from left_right_window import left_right
from finger_movement import find_movement
from iris_x_exercise import iris_x_exercise
from blink_exercixe import blink
import cv2

def main():
#     print("Select a mode:")
#     print("1 - Forward/Backward Tracking")
#     print("2 - Left/Right Tracking")
#     print("3 - Pose")

#     choice = input("Enter your choice: ")

#     if choice =="1":
#        forward_backward()
#     elif choice =="2":
#         left_right()
#     elif choice =="3":
#         find_movement()
    
#     else:
#         print("Invalid choice.")
    exercises = [
        # iris_x_exercise(),
        # blink(),
        # forward_back(),
        # left_right(),
        find_movement()
    ]

    cap = cv2.VideoCapture(0)

    current = 0

    while cap.isOpened():
        ret, frame1 = cap.read()
        if not ret: break
        frame = cv2.flip(frame1, 1)
        frame, done = exercises[current].process_frame(frame)

        cv2.putText(frame, f"Exercise {current+1}/{len(exercises)}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Workout", frame)

        if done:
            current += 1
            if current == len(exercises):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ =="__main__":
    main()