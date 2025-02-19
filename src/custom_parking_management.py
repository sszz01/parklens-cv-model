import cv2
import json
import numpy as np
from shapely.geometry import Polygon
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

def calculate_iou(park_spot_polygon, vehicle_bbox):
    # Create a Polygon for the vehicle's bounding box
    vehicle_polygon = Polygon([(vehicle_bbox[0], vehicle_bbox[1]),
                               (vehicle_bbox[0] + vehicle_bbox[2], vehicle_bbox[1]),
                               (vehicle_bbox[0] + vehicle_bbox[2], vehicle_bbox[1] + vehicle_bbox[3]),
                               (vehicle_bbox[0], vehicle_bbox[1] + vehicle_bbox[3])])

    # Calculate the intersection and union of the parking spot and vehicle polygon
    intersection_area = park_spot_polygon.intersection(vehicle_polygon).area
    union_area = park_spot_polygon.union(vehicle_polygon).area

    # Return the Intersection over Union (IoU)
    if union_area == 0:
        return 0  # Avoid division by zero
    return intersection_area / union_area

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

        self.pr_info = {"Occupied": 0, "Available": len(self.json)}
        self.motion_tracker = motion_tracker
        self.arc = COLOR_GREEN
        self.occ = COLOR_RED

    def process_data(self, frame):
        self.extract_tracks(frame)
        total_spots = len(self.json)
        occupied_spots = 0
        detected_vehicles = []
        for obj_list in self.motion_tracker(frame):
            detected_vehicles.extend(obj_list)
        annotator = Annotator(frame, self.line_width)

        for region in self.json:
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            park_spot_polygon = Polygon([(pt[0], pt[1]) for pt in region["points"]])
            is_occupied = False
            for (x, y, w, h) in detected_vehicles:
                vehicle_bbox = (x, y, w, h)
                iou = calculate_iou(park_spot_polygon, vehicle_bbox)
                if iou > 0.25:
                    is_occupied = True
                    break
            cv2.polylines(frame, [pts_array], isClosed=True, color=self.occ if is_occupied else self.arc, thickness=3)
            if is_occupied:
                occupied_spots += 1
        self.pr_info["Occupancy"] = occupied_spots
        self.pr_info["Available"] = total_spots - occupied_spots
        annotator.display_analytics(frame, self.pr_info, (104, 31, 17), (255, 255, 255), 10)

        return frame
