from ultralytics.solutions import ParkingPtsSelection

class CustomParkingPtsSelection(ParkingPtsSelection):
    def __init__(self, canvas_name, canvas_width, canvas_height):
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.master = self.tk.Tk()
        self.master.title(canvas_name)  # Override window title
        self.master.resizable(False, False)

        # Override canvas size before initialization
        self.canvas_max_width, self.canvas_max_height = canvas_width, canvas_height

        self.canvas = self.tk.Canvas(self.master, bg="white", width=self.canvas_max_width,
                                     height=self.canvas_max_height)
        self.canvas.pack(side=self.tk.BOTTOM)

        self.image = None
        self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0

        # Button frame
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

        self.master.mainloop()