from forward_backward_window import forward_backward
from left_right_window import left_right
from finger_movement import find_movement

def main():
    print("Select a mode:")
    print("1 - Forward/Backward Tracking")
    print("2 - Left/Right Tracking")
    print("3 - Pose")

    choice = input("Enter your choice: ")

    if choice =="1":
       forward_backward()
    elif choice =="2":
        left_right()
    elif choice =="3":
        find_movement()
    
    else:
        print("Invalid choice.")



if __name__ =="__main__":
    main()