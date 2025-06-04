import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(0.1*width), height),
        (int(0.4*width), int(0.6*height)),
        (int(0.6*width), int(0.6*height)),
        (int(0.9*width), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts the frame to grayscale        
    blur = cv2.GaussianBlur(gray, (5, 5), 0)    # Applies a Gaussian blur to the frame
    edges = cv2.Canny(blur, 50, 150)    # Detects edges in the frame
    cropped_edges = region_of_interest(edges)    # Crops the frame to the region of interest

    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
    lane_img = frame.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_img, (x1, y1), (x2, y2), (255, 255, 0), 4)

    return lane_img
