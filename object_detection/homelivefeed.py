import cv2
from ultralytics import YOLO
import subprocess
import os
import time

def main():
    # Use your actual IP camera stream URL here
    ip_camera_url = "rtsp://username:password@ipaddress:port/stream"

    cap = cv2.VideoCapture(ip_camera_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    model = YOLO("yolov8n.pt")

    # Use absolute path for alert.wav to avoid path issues
    alert_sound = os.path.abspath("alert.wav")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        result = model.predict(source=frame, show=False, conf=0.5)
        results = result[0]
        detected_frame = results.plot()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        if 0 in classes:
            print("person detected")
            cv2.imwrite("person.jpg", frame)
            # Play alert sound using afplay via subprocess (macOS)
            subprocess.call(['afplay', alert_sound])
            time.sleep(1)  # Prevent multiple alerts per second

        cv2.imshow("Live Object Detection from CCTV Feed", detected_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
