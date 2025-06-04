import cv2
from ultralytics import YOLO
from lane_detection import detect_lane

def main():
    cap = cv2.VideoCapture("nD_11.mp4")
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    model = YOLO("yolov8n.pt")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lane detection
        lane_frame = detect_lane(frame)

        # Object detection
        results = model.predict(source=lane_frame, show=False, conf=0.7)
        final_frame = results[0].plot()

        cv2.imshow("Lane + Object Detection", final_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
