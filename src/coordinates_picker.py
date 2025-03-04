import json
import os

import cv2
from ultralytics.solutions import ParkingPtsSelection

class CustomParkingPtsSelection(ParkingPtsSelection):
    def __init__(self, canvas_name, canvas_width, canvas_height, first_frame=None, polygon_json_path=None):
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.master = self.tk.Tk()
        self.master.title(canvas_name)
        self.master.resizable(False, False)
        self.canvas_max_width, self.canvas_max_height = canvas_width, canvas_height
        self.canvas = self.tk.Canvas(self.master, bg="white", width=self.canvas_max_width,
                                     height=self.canvas_max_height)
        self.canvas.pack(side=self.tk.BOTTOM)
        self.image = None
        self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.polygon_json_path = polygon_json_path

        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.custom_save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

        if first_frame is not None:
            self.upload_image_from_frame(first_frame)

        self.master.mainloop()

    def upload_image_from_frame(self, first_frame):
        """Loads the first frame from the video and displays it on the canvas."""
        from PIL import Image, ImageTk  # scope because ImageTk requires tkinter package

        self.image = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height)))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def custom_save_to_json(self):
        """Saves the bounding box data to a JSON file, merging with existing data."""
        if not self.rg_data:
            self.messagebox.showinfo("Info", "No bounding boxes to save.")
            return

        # Load existing bboxes if the file exists
        if os.path.exists(self.polygon_json_path):
            with open(self.polygon_json_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend([{"points": bbox} for bbox in self.rg_data])
        with open(self.polygon_json_path, "w") as f:
            json.dump(existing_data, f, indent=4)

        self.messagebox.showinfo("Success", "Bounding boxes saved.")