import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

# کلاس‌های تمرین که هر کدوم process_frame(self, frame) دارند
from forward_backward_window import forward_back
from left_right_window import left_right
from finger_movement import find_movement
from iris_x_exercise import iris_x_exercise
from blink_exercixe import blink


# ---------------- تم آبی نرم ----------------
def apply_theme(root: tk.Tk) -> ttk.Style:
    BG = "#F7FAFF"
    CARD = "#FFFFFF"
    TEXT = "#0B1B3A"
    MUTED = "#5B6B86"
    BLUE = "#2563EB"
    BLUE_D = "#1D4ED8"
    BLUE_P = "#1E40AF"
    SOFT = "#EAF1FF"
    DANGER_BG = "#FFE8E8"
    DANGER_TXT = "#7A1222"
    WELCOME_BG = "#E3EEFF"  # پس‌زمینه نرم برای ولکام و فینیش

    root.configure(bg=BG)
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", font=("Segoe UI", 11))
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)

    # کارت‌ها
    style.configure("Card.TFrame", background=CARD)
    style.configure("Card.TLabel", background=CARD, foreground=TEXT)

    # صفحه‌های ولکام و فینیش
    style.configure("Welcome.TFrame", background=WELCOME_BG)
    style.configure("Finish.TFrame", background=WELCOME_BG)

    # تایپوگرافی
    style.configure("AppTitle.TLabel", font=("Segoe UI", 22, "bold"),
                    foreground=TEXT, background=BG)
    style.configure("ExerciseTitle.TLabel", font=("Segoe UI", 15, "bold"),
                    foreground=TEXT, background=CARD)
    style.configure("Hint.TLabel", font=("Segoe UI", 11),
                    foreground="#1E3A8A", background=CARD)
    style.configure("Meta.TLabel", font=("Segoe UI", 10),
                    foreground=MUTED, background=CARD)
    style.configure("Status.TLabel", font=("Segoe UI", 10),
                    foreground=MUTED, background=BG)

    # دکمه‌ها
    style.configure(
        "Blue.TButton",
        background=BLUE,
        foreground="white",
        padding=(18, 10),
        borderwidth=0,
        focusthickness=0,
        focuscolor="none",
    )
    style.map(
        "Blue.TButton",
        background=[("active", BLUE_D), ("pressed", BLUE_P), ("disabled", "#A7B7E8")],
        foreground=[("disabled", "#EEF2FF")],
    )

    style.configure(
        "Soft.TButton",
        background=SOFT,
        foreground=TEXT,
        padding=(16, 9),
        borderwidth=0,
        focusthickness=0,
        focuscolor="none",
    )
    style.map("Soft.TButton", background=[("active", "#DDE9FF"), ("pressed", "#CFE0FF")])

    style.configure(
        "Danger.TButton",
        background=DANGER_BG,
        foreground=DANGER_TXT,
        padding=(16, 9),
        borderwidth=0,
        focusthickness=0,
        focuscolor="none",
    )
    style.map("Danger.TButton", background=[("active", "#FFDADA"), ("pressed", "#FFC6C6")])

    # نوار پیشرفت
    style.configure(
        "Blue.Horizontal.TProgressbar",
        troughcolor=SOFT,
        background=BLUE,
        thickness=12,
        borderwidth=0,
    )

    return style
def add_soft_pattern_bg(parent, base="#E3EEFF"):
    """افزودن بک‌گرند پترن نرم با چند بیضی آبی روشن"""
    canvas = tk.Canvas(parent, bd=0, highlightthickness=0, bg=base)
    canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

    def redraw(event):
        canvas.delete("all")
        w, h = event.width, event.height

        # چند تا بیضی خیلی نرم گوشه‌ها
        canvas.create_oval(
            -int(w * 0.3), int(h * 0.05),
            int(w * 0.5), int(h * 0.7),
            fill="#D7E5FF", outline=""
        )
        canvas.create_oval(
            int(w * 0.4), -int(h * 0.25),
            int(w * 1.1), int(h * 0.5),
            fill="#C4D6FF", outline=""
        )
        canvas.create_oval(
            int(w * 0.2), int(h * 0.55),
            int(w * 0.9), int(h * 1.2),
            fill=base, outline=""
        )

    canvas.bind("<Configure>", redraw)
    return canvas

# ---------------- کلاس اصلی برنامه ----------------
class EyeGuardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        apply_theme(root)

        self.root.title("EyeGuard")
        self.root.geometry("980x720")
        self.root.minsize(880, 660)

        # وضعیت اجرا
        self.cap = None
        self.running = False
        self._after_id = None
        self._pending_next = False
        self.current_exercise_index = 0

        # متن‌های فارسی تمرین‌ها
        self.titles = [
            "تمرین: حرکت چشم (چپ/راست)",
            "تمرین: پلک زدن",
            "تمرین: گردن (جلو/عقب)",
            "تمرین: گردن (چپ/راست)",
            "تمرین: حرکت انگشت‌ها",
        ]
        self.hints = [
            "نگاهت را آرام به چپ و راست ببر.",
            "۱۰ بار آرام پلک بزن.",
            "سر را بدون فشار، آرام جلو و عقب ببر.",
            "سر را آرام به چپ و راست بچرخان.",
            "انگشت‌ها را چند بار باز و بسته کن (هر دو دست).",
        ]
        self.targets = [5, 10, 5, 5, 10]

        # کلاس‌های تمرین (آبجکت‌ها بعداً ساخته می‌شن)
        self.exercise_classes = [iris_x_exercise, blink, forward_back,
                                 left_right, find_movement]
        self.exercise_objects = []

        # سه صفحه اصلی
        self.main_page = ttk.Frame(root, padding=22)
        self.welcome_page = ttk.Frame(root, padding=30, style="Welcome.TFrame")
        self.finish_page = ttk.Frame(root, padding=30, style="Finish.TFrame")

        self._build_main_ui()
        self._build_welcome_ui()
        self._build_finish_ui()

        self._show_welcome()

    # ---------- توابع کمکی ----------
    def fa_num(self, n: int) -> str:
        return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))

    def _cancel_after(self):
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _schedule_next_frame(self, delay_ms: int = 20):
        self._cancel_after()
        self._after_id = self.root.after(delay_ms, self.update_frame)

    def _release_camera(self):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass

    def ensure_camera(self) -> bool:
        """باز کردن دوربین (سازگار با ویندوز)."""
        if self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.status_bar.config(
                text="❌ دوربین باز نشد. احتمالاً یک برنامه‌ی دیگر از دوربین استفاده می‌کند."
            )
            return False

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception:
            pass

        return True

    def _build_exercise_objects(self):
        self.exercise_objects = [cls() for cls in self.exercise_classes]

    # ---------- جابه‌جایی بین صفحات ----------
    def _hide_all_pages(self):
        for p in (self.main_page, self.welcome_page, self.finish_page):
            try:
                p.pack_forget()
            except Exception:
                pass

    def _show_main(self):
        self._hide_all_pages()
        self.main_page.pack(fill="both", expand=True)

    def _show_welcome(self):
        self._hide_all_pages()
        self.welcome_page.pack(fill="both", expand=True)

    def _show_finish(self):
        self._hide_all_pages()
        self.finish_page.pack(fill="both", expand=True)

    # ---------- UI اصلی با grid ----------
    def _build_main_ui(self):
        # grid برای صفحه اصلی
        self.main_page.grid_rowconfigure(0, weight=0)  # header
        self.main_page.grid_rowconfigure(1, weight=0)  # info
        self.main_page.grid_rowconfigure(2, weight=1)  # video
        self.main_page.grid_rowconfigure(3, weight=0)  # buttons
        self.main_page.grid_rowconfigure(4, weight=0)  # status
        self.main_page.grid_columnconfigure(0, weight=1)

        # ردیف ۰: هدر
        header = ttk.Frame(self.main_page)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(header, text="EyeGuard", style="AppTitle.TLabel").pack(
            side="left", anchor="w"
        )

        # ردیف ۱: کارت اطلاعات تمرین
        info_card = ttk.Frame(self.main_page, style="Card.TFrame", padding=14)
        info_card.grid(row=1, column=0, sticky="ew", pady=(4, 12))

        self.exercise_label = ttk.Label(
            info_card,
            text="",
            style="ExerciseTitle.TLabel",
            anchor="e",
            justify="right",
        )
        self.exercise_label.pack(fill="x")

        self.hint_label = ttk.Label(
            info_card,
            text="",
            style="Hint.TLabel",
            anchor="e",
            justify="right",
            wraplength=880,
        )
        self.hint_label.pack(fill="x", pady=(4, 10))

        self.progress = ttk.Progressbar(
            info_card,
            style="Blue.Horizontal.TProgressbar",
            maximum=100,
            value=0,
        )
        self.progress.pack(fill="x")

        self.counter_label = ttk.Label(
            info_card,
            text="",
            style="Meta.TLabel",
            anchor="e",
            justify="right",
        )
        self.counter_label.pack(fill="x", pady=(6, 0))

        # ردیف ۲: کارت وبکم (وسط صفحه)
        video_card = ttk.Frame(self.main_page, style="Card.TFrame", padding=12)
        video_card.grid(row=2, column=0, sticky="nsew", pady=(0, 10))

        video_card.grid_rowconfigure(0, weight=1)
        video_card.grid_columnconfigure(0, weight=1)

        video_inner = ttk.Frame(video_card, style="Card.TFrame")
        video_inner.grid(row=0, column=0, sticky="nsew")
        video_inner.grid_rowconfigure(0, weight=1)
        video_inner.grid_columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_inner, style="Card.TLabel")
        self.video_label.grid(row=0, column=0)  # وسط سلول

        # ردیف ۳: دکمه‌ها (ثابت، همیشه دیده می‌شن)
        controls = ttk.Frame(self.main_page)
        controls.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        self.btn_skip = ttk.Button(
            controls,
            text="رد کردن تمرین",
            style="Soft.TButton",
            command=self.skip_exercise,
        )
        self.btn_skip.pack(side="left")

        self.btn_exit = ttk.Button(
            controls,
            text="خروج",
            style="Danger.TButton",
            command=self.close,
        )
        self.btn_exit.pack(side="right")

        # ردیف ۴: نوار وضعیت
        self.status_bar = ttk.Label(
            self.main_page,
            text="",
            style="Status.TLabel",
            anchor="w",
            justify="left",
        )
        self.status_bar.grid(row=4, column=0, sticky="ew", pady=(6, 0))

    # ---------- صفحه خوش‌آمد ----------
    def _build_welcome_ui(self):
            # پس‌زمینه‌ی پترن‌دار
        add_soft_pattern_bg(self.welcome_page, base="#E3EEFF")
        # فقط کارت وسط، پس‌زمینه‌ی کل صفحه آبی نرم
        card = ttk.Frame(self.welcome_page, style="Card.TFrame", padding=26)
        card.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(
            card,
            text="خوش اومدی به EyeGuard 👋",
            style="ExerciseTitle.TLabel",
            anchor="center",
        ).pack(pady=(0, 10))

        ttk.Label(
            card,
            text=(
                "چند دقیقه مراقبت، کلی حال بهتر ✨\n\n"
                "این برنامه در فواصل زمانی مشخص، تمرین‌های کوتاه برای چشم، گردن و دست‌ها "
                "بهت یادآوری می‌کند و اجرای آن‌ها را با وب‌کم پایش می‌کند.\n\n"
                "وقتی آماده‌ای، روی دکمهٔ زیر بزن و اولین جلسه را شروع کن."
            ),
            style="Meta.TLabel",
            justify="center",
            anchor="center",
            wraplength=520,
        ).pack(pady=(0, 18))

        ttk.Button(
            card, text="شروع تمرین", style="Blue.TButton", command=self.start
        ).pack()

    # ---------- صفحه پایان ----------
    def _build_finish_ui(self):
        add_soft_pattern_bg(self.finish_page, base="#E3EEFF")

        card = ttk.Frame(self.finish_page, style="Card.TFrame", padding=26)
        card.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(card, text="آفرین! 🎉", style="ExerciseTitle.TLabel").pack(
            pady=(0, 10)
        )

        ttk.Label(
            card,
            text=(
                "تمرین‌های این جلسه تموم شد.\n"
                "اگر دوست داشتی می‌تونی دوباره از اول شروع کنی یا از برنامه خارج بشی."
            ),
            style="Meta.TLabel",
            justify="center",
            wraplength=480,
        ).pack(pady=(0, 18))

        row = ttk.Frame(card, style="Card.TFrame")
        row.pack()

        ttk.Button(
            row, text="شروع مجدد", style="Blue.TButton", command=self.restart
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            row, text="خروج", style="Soft.TButton", command=self.close
        ).pack(side="left")

    # ---------- به‌روزرسانی متن تمرین ----------
    def update_exercise_ui(self, count: int, done: bool):
        title = self.titles[self.current_exercise_index]
        hint = self.hints[self.current_exercise_index]
        target = self.targets[self.current_exercise_index]

        self.exercise_label.config(text=title)
        self.hint_label.config(text=f"راهنما: {hint}")
        self.counter_label.config(
            text=f"تکرار: {self.fa_num(count)} از {self.fa_num(target)}"
        )

        pct = 0 if target == 0 else int((count / target) * 100)
        self.progress.configure(value=max(0, min(100, pct)))

        if done:
            self.status_bar.config(text="✅ انجام شد! می‌ریم سراغ تمرین بعدی…")

    # ---------- کنترل‌ها ----------
    def start(self):
        if self.running:
            return

        self._show_main()

        if not self.ensure_camera():
            self._show_welcome()
            return

        self._build_exercise_objects()

        self.running = True
        self._pending_next = False
        self.current_exercise_index = 0

        self.progress.configure(value=0)
        self.status_bar.config(text="در حال اجرا…")
        self.update_exercise_ui(count=0, done=False)

        self._schedule_next_frame(0)

    def restart(self):
        self.running = False
        self._cancel_after()
        self._release_camera()

        self.current_exercise_index = 0
        self._pending_next = False
        self.progress.configure(value=0)
        self.status_bar.config(text="")

        self.video_label.configure(image="")
        self.video_label.imgtk = None

        self._show_welcome()

    def skip_exercise(self):
        if not self.running or self._pending_next:
            return
        self._pending_next = True
        self.root.after(1, self._go_next)

    def _go_next(self):
        self.current_exercise_index += 1

        if self.current_exercise_index >= len(self.exercise_objects):
            # همه تمرین‌ها تموم شد
            self.running = False
            self._cancel_after()
            self._release_camera()

            self.video_label.configure(image="")
            self.video_label.imgtk = None

            self.current_exercise_index = 0
            self._pending_next = False
            self._show_finish()
            return

        self._pending_next = False
        self.status_bar.config(text="تمرین بعدی آماده است.")
        self.update_exercise_ui(count=0, done=False)

    # ---------- حلقه ویدیو ----------
    def update_frame(self):
        self._after_id = None

        if not self.running or self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self._schedule_next_frame(30)
            return

        frame = cv2.flip(frame, 1)

        exercise = self.exercise_objects[self.current_exercise_index]
        title = self.titles[self.current_exercise_index]

        try:
            processed_frame, done, count = exercise.process_frame(frame)
        except Exception as e:
            self.status_bar.config(text=f"❌ خطا در «{title}»: {e}")
            processed_frame, done, count = frame, False, 0

        self.update_exercise_ui(count=count, done=done)

        if done and not self._pending_next:
            self._pending_next = True
            self.root.after(700, self._go_next)

        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        w = self.video_label.winfo_width()
        h = self.video_label.winfo_height()
        if w > 50 and h > 50:
            img.thumbnail((w, h))

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self._schedule_next_frame(20)

    def close(self):
        self.running = False
        self._cancel_after()
        self._release_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = EyeGuardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()