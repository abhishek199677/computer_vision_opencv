import cv2
from ultralytics import YOLO
from playsound import playsound  # Use playsound instead of winsound for macOS

def main():
    # Use your actual IP camera stream URL here
    ip_camera_url = "username/password/ipaddress:port"

    

    cap = cv2.VideoCapture(ip_camera_url)
    
    model = YOLO("yolov8n.pt")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        result = model.predict(source=frame, show=False, conf=0.5)
        results = result[0]
        detected_frame = result[0].plot()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        if 0 in classes:
            print("person detected")
            cv2.imwrite("person.jpg", frame)
            playsound('alert.wav')  # Use a valid sound file path
            
        cv2.imshow("live object detection from my CCTV feed", detected_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
