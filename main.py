import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

# --- Import exercises ---
from forward_backward_window import forward_back
from left_right_window import left_right
from finger_movement import find_movement
from iris_x_exercise import iris_x_exercise
from blink_exercixe import blink

# ---------- Rounded Progressbar ----------
class RoundedProgressbar(tk.Canvas):
    def __init__(self, parent, width=400, height=20, bg="#2c2c2c", fg="#00d47e", max_value=100, **kwargs):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.fg = fg
        self.bg = bg
        self.max_value = max_value
        self.value = 0
        self.rounded_rect(0, 0, width, height, radius=10, fill=bg)

    def rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        points = [x1+radius, y1,
                  x2-radius, y1,
                  x2, y1, x2, y1+radius,
                  x2, y2-radius, x2, y2,
                  x2-radius, y2,
                  x1+radius, y2,
                  x1, y2, x1, y2-radius,
                  x1, y1+radius,
                  x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def set(self, value):
        self.value = min(max(value, 0), self.max_value)
        self.delete("bar")
        fill_width = (self.value/self.max_value)*self.width
        self.rounded_rect(0, 0, fill_width, self.height, radius=10, fill=self.fg, tags="bar")

# ---------- Main App ----------
class EyeGuardApp:
    def __init__(self, root):
        self.exercise_tips = [
            "چشم‌ها را به سمت چپ و راست حرکت دهید",
            "چشم‌ها را چند بار سریع ببندید و باز کنید",
            "سر را جلو و عقب حرکت دهید",
            "سر را به چپ و راست بچرخانید",
            "انگشتان دست خود را تکان دهید"
        ]

        self.root = root
        self.root.title("EyeGuard Trainer")
        self.root.geometry("1000x720")
        self.root.configure(bg="#1e1e1e")

        # ---------- Styles ----------
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton",
                        font=("Poppins", 14),
                        padding=10,
                        background="#00d47e",
                        foreground="white",
                        borderwidth=0)
        style.map("TButton", background=[("active", "#00b86b")])

        # ---------- UI Layout ----------
        self.title_label = tk.Label(root,
                                    text="EyeGuard Exercise Program",
                                    font=("Poppins", 26, "bold"),
                                    fg="white",
                                    bg="#1e1e1e")
        self.title_label.pack(pady=10)

        self.exercise_label = tk.Label(root,
                                       text="Press Start",
                                       font=("Poppins", 18),
                                       fg="#00d47e",
                                       bg="#1e1e1e")
        self.exercise_label.pack()

        self.tip_label = tk.Label(root,
                                  text="",
                                  font=("Poppins", 16),
                                  fg="yellow",
                                  bg="#1e1e1e",
                                  wraplength=800,
                                  justify="center")
        self.tip_label.pack(pady=5)

        self.count_label = tk.Label(root,
                                    text="",
                                    font=("Poppins", 16),
                                    fg="#00d47e",
                                    bg="#1e1e1e")
        self.count_label.pack(pady=5)

        self.progress = RoundedProgressbar(root, width=400, height=20)
        self.progress.pack(pady=10)

        self.video_label = tk.Label(root, bg="#2c2c2c")
        self.video_label.pack(pady=15)

        button_frame = tk.Frame(root, bg="#1e1e1e")
        button_frame.pack(pady=20)

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start)
        self.start_button.grid(row=0, column=0, padx=10)

        self.restart_button = ttk.Button(button_frame, text="Restart", command=self.restart)
        self.restart_button.grid(row=0, column=1, padx=10)

        self.exit_button = ttk.Button(button_frame, text="Exit", command=self.quit_app)
        self.exit_button.grid(row=0, column=2, padx=10)

        # ---------- Variables ----------
        self.cap = None
        self.exercises = []
        self.current_exercise_index = 0
        self.running = False

    # ------------------ START ------------------ #
    def start(self):
        self.exercises = [
            iris_x_exercise(),
            blink(),
            forward_back(),
            left_right(),
            find_movement()
        ]
        self.current_exercise_index = 0
        self.running = True

        self.cap = cv2.VideoCapture(0)
        self.progress.set(0)
        self.count_label.config(text="")  # شروع بدون count

        self.update_frame()

    # ------------------ RESTART ------------------ #
    def restart(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        self.current_exercise_index = 0
        self.progress.set(0)
        self.exercise_label.config(text="Press Start")
        self.tip_label.config(text="")
        self.count_label.config(text="")
        self.video_label.config(image="")

    # ------------------ EXIT ------------------ #
    def quit_app(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.root.destroy()

    # ------------------ UPDATE FRAME ------------------ #
    def update_frame(self):
        if not self.running or self.current_exercise_index >= len(self.exercises):
            # پایان تمرین‌ها → آزادسازی وب‌کم
            if self.cap:
                self.cap.release()
                self.cap = None
            self.exercise_label.config(text="Workout Complete!")
            self.tip_label.config(text="")
            self.count_label.config(text="")
            self.video_label.config(image="")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        exercise = self.exercises[self.current_exercise_index]

        # پردازش تمرین
        frame, done, count = exercise.process_frame(frame)

        # Update UI
        self.exercise_label.config(
            text=f"Exercise {self.current_exercise_index + 1}/{len(self.exercises)}"
        )
        self.tip_label.config(text=self.exercise_tips[self.current_exercise_index])
        self.count_label.config(text=f"Count: {count}")

        progress_value = int(((self.current_exercise_index + (1 if done else 0)) / len(self.exercises)) * 100)
        self.progress.set(progress_value)

        # نمایش فریم
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Move to next exercise if done
        if done:
            self.current_exercise_index += 1

        self.root.after(10, self.update_frame)


# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeGuardApp(root)
    root.mainloop()
