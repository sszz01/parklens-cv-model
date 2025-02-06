import cv2
import numpy as np
from ultralytics.solutions import ParkingManagement
from data.colors import *
# from ultralytics.utils.plotting import Annotator

class CustomParkingManagement(ParkingManagement):
    def __init__(self, motion_tracker, **kwargs):
        """
        Custom Parking Management that integrates motion tracking for better parking occupancy detection.
        """
        super().__init__(**kwargs)
        self.motion_tracker = motion_tracker  # store motion tracker function
        self.arc = COLOR_GREEN
        self.occ = COLOR_RED

    def process_data(self, frame):
        """
        Overrides the default process_data method to improve motion tracking.
        """
        self.extract_tracks(frame)
        es, fs = len(self.json), 0
        cars, motorcycles, buses, trucks = self.motion_tracker(frame)
        detected_vehicles = cars + motorcycles + buses + trucks

        for region in self.json:
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False
            for (x, y, w, h) in detected_vehicles:
                xc, yc = x + w // 2, y + h // 2
                dist = cv2.pointPolygonTest(pts_array, (xc, yc), False)
                if dist >= 0:  # if a vehicle is inside a parking region
                    rg_occupied = True
                    break
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
            cv2.polylines(frame, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=3)
        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, es
        self.display_output(frame)
        return frame
