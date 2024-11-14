import cv2
import pickle

cap = cv2.VideoCapture("media/parking-lot.mp4") # select a video
# success,frame=cap.read()
# frame=cv2.resize(frame,(640,480))
# roi=cv2.selectROI(frame)
try:
    with open("cparkyt", 'rb') as f:
        points = pickle.load(f)
except:
    points = []


def drawing(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        with open("cparkyt", 'wb') as f:
            pickle.dump(points, f)
    if event == cv2.EVENT_RBUTTONDOWN:
        for i, pts in enumerate(points):
            x1, y1 = pts
            if x1 < x + w and y1 < y + h:
                points.pop(i)


w, h = 29, 27
while True:
    success, frame = cap.read()
    if success == False:
        break

    frame = cv2.resize(frame, (640, 480))
    for pts in points:
        cv2.rectangle(frame, pts, (pts[0] + w, pts[1] + h), (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", drawing)
    if cv2.waitKey(32) == ord("q"): # quit when Q is pressed
        break
cap.release()
cv2.destroyAllWindows()