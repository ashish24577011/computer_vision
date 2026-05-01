import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch

model = torch.hub.load(
    'WongKinYiu/yolov7',
    'custom',
    'yolov7-tiny.pt',
    trust_repo=True
)

model.conf = 0.4
class_names = model.names

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv7 Tiny Object Detection")
        self.root.geometry("900x700")

        self.video_label = Label(root)
        self.video_label.pack()

        self.start_button = Button(
            root,
            text="Start Camera",
            command=self.start_camera,
            font=("Arial", 14),
            bg="green",
            fg="white"
        )
        self.start_button.pack(pady=10)

        self.stop_button = Button(
            root,
            text="Stop Camera",
            command=self.stop_camera,
            font=("Arial", 14),
            bg="red",
            fg="white"
        )
        self.stop_button.pack(pady=10)

        self.cap = None
        self.running = False

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    def update_frame(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()

            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(rgb_frame)
                detections = results.xyxy[0].cpu().numpy()

                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    label = f"{class_names[int(cls)]} {conf:.2f}"

                    cv2.rectangle(
                        rgb_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        rgb_frame,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

                resized_frame = cv2.resize(rgb_frame, (850, 600))
                img = Image.fromarray(resized_frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()