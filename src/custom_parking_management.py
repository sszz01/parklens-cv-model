import cv2
import json
import numpy as np
from ultralytics.solutions import ParkingManagement
from data.colors import *
from ultralytics.utils.plotting import Annotator


def draw_vehicles(motion_tracker, frame):
    cars, motorcycles, buses, trucks = motion_tracker(frame)
    for car in cars:
        x, y, w, h = car
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_RED, 2)
        cv2.putText(frame, "Car", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEAL, 2)

    for motorcycle in motorcycles:
        x, y, w, h = motorcycle
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BLUE, 2)
        cv2.putText(frame, "Motorcycle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)

    for bus in buses:
        x, y, w, h = bus
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)
        cv2.putText(frame, "Bus", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_MAGENTA, 2)

    for truck in trucks:
        x, y, w, h = truck
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_ORANGE, 2)
        cv2.putText(frame, "Truck", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)
    return frame


class CustomParkingManagement(ParkingManagement):
    def __init__(self, motion_tracker, **kwargs):
        """
        Custom Parking Management that integrates motion tracking for better parking occupancy detection.
        """
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]
        if self.json_file is None:
            raise ValueError("❌ Json file path cannot be empty")

        with open(self.json_file) as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": len(self.json)}  # Start with all available
        self.motion_tracker = motion_tracker
        self.arc = COLOR_GREEN
        self.occ = COLOR_RED

    def process_data(self, frame):
        self.extract_tracks(frame)
        total_spots = len(self.json)
        occupied_spots = 0
        detected_vehicles = sum(self.motion_tracker(frame), [])
        annotator = Annotator(frame, self.line_width)

        frame = draw_vehicles(self.motion_tracker, frame)

        for region in self.json:
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            is_occupied = False
            for (x, y, w, h) in detected_vehicles:
                xc, yc = x + w // 2, y + h // 2
                if cv2.pointPolygonTest(pts_array, (xc, yc), False) >= 0:
                    annotator.display_objects_labels(frame, "Vehicle", (104, 31, 17), (255, 255, 255), xc, yc, 10)
                    is_occupied = True
                    break
            cv2.polylines(frame, [pts_array], isClosed=True, color=self.occ if is_occupied else self.arc, thickness=3)
            if is_occupied:
                occupied_spots += 1
        self.pr_info["Occupancy"] = occupied_spots
        self.pr_info["Available"] = total_spots - occupied_spots
        annotator.display_analytics(frame, self.pr_info, (104, 31, 17), (255, 255, 255), 10)

        return frame
