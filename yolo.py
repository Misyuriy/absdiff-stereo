from ultralytics import YOLO
import cv2


model = YOLO('yolov8n.pt')  # автоматически скачает модель


def process_video(input_path):
    cap = cv2.VideoCapture(input_path)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        results = model.track(frame, persist=True)

        annotated_frame = results[0].plot()

        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video("videos/cam3/cam32.mp4")
