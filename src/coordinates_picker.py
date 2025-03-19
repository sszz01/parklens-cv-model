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
        self.polygon_json_path = polygon_json_path if polygon_json_path else "bounding_boxes.json"

        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        self.save_button = self.tk.Button(button_frame, text="Save", command=self.custom_save_to_json)
        self.save_button.pack(side=self.tk.LEFT)

        for text, cmd in [
            ("Remove All BBoxes", self.remove_all_bounding_boxes)
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

        if first_frame is not None:
            self.upload_image_from_frame(first_frame)

        if os.path.exists(self.polygon_json_path):
            with open(self.polygon_json_path, "r") as f:
                self.rg_data = json.load(f)
        else:
            self.rg_data = []

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
        """Saves the bounding box data to a JSON file, ensuring no duplication and fixing the nested structure."""
        if not self.rg_data:
            self.messagebox.showinfo("Info", "No bounding boxes to save.")
            return

        # Ensure rg_data is a list of dictionaries with "points" key
        try:
            new_bbox_data = [{"points": bbox} if isinstance(bbox, list) else bbox for bbox in self.rg_data]
        except Exception as e:
            self.messagebox.showerror("Error", f"Invalid bounding box format: {e}")
            return

        # Check if the new data is the same as the current saved data
        try:
            with open(self.polygon_json_path, "r") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        if new_bbox_data == existing_data:
            self.messagebox.showinfo("Info", "No changes to save.")
            return

        with open(self.polygon_json_path, "w") as f:
            json.dump(new_bbox_data, f, indent=4)

        self.messagebox.showinfo("Success", "Bounding boxes saved.")

    def remove_all_bounding_boxes(self):
        if not self.rg_data:
            self.messagebox.showinfo("Info", "No bounding boxes to remove.")
            return
        self.rg_data.clear()

        self.save_button.config(state=self.tk.NORMAL)

        with open(self.polygon_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.messagebox.showinfo("Success", "All bounding boxes removed.")