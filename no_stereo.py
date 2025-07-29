import numpy as np
import cv2 as cv

from detection import detect_objects
from opencv_utils import draw_text_lines


if __name__ == "__main__":
    file_name = "C:/Users/andre/PycharmProjects/turret/videos/cam0/cam01.mp4"
    upside_down: bool = False # некоторые видео открывает перевернутыми
    
    cap = cv.VideoCapture(file_name)

    cv.namedWindow("resized", cv.WINDOW_NORMAL)
    cv.setWindowProperty("resized", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty("resized", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    (x, y, width, height) = cv.getWindowImageRect("resized")

    border = 50

    ret, frame = cap.read()

    f = min((float(width) / (float(frame.shape[1]) - 2 * border)),
            (float(height) / (float(frame.shape[0]) - 2 * border)))
    cv.resizeWindow("resized", round(float(frame.shape[1] - 2 * border) * f),
                    round(float(frame.shape[0] - 2 * border) * f))

    is_first_resize = True
    paused = False

    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)


    small_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    large_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

        f = min((float(width) / (float(frame.shape[1] / 2) - 2 * border)),
                (float(height) / (float(frame.shape[0]) - 2 * border)))
        
        # detection/tracking
        boxes = []
        if not paused: # не пихать в историю одинаковые бессмысленные кадры на паузе
            bg_mask = bg_subtractor.apply(frame)

            bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_OPEN, small_kernel)
            #bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_CLOSE, large_kernel)

            boxes = detect_objects(bg_mask, min_area=100, max_area=500)


        # рисуем bounding boxes и пишем информацию об объекте из карты глубины

        for (x, y, w, h, a) in boxes:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            #cv.putText(frame, f"Detected", (x, y - 20), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        status_text = ""
        if paused:
            status_text = "Paused"
        else:
            status_text = f"Detecting"

        cv.putText(frame, status_text, (32, 32),
                   cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

        controls_text = ["SPACE - pause",
                         "Q - quit"]

        draw_text_lines(frame, controls_text, (32, height - 64), bottom_origin=True, color=(0, 255, 255))

        cv.imshow("resized", frame)

        key = cv.waitKey(10)

        if key == ord(" "):  # pause/play
            paused = not paused
        if key == ord("q"):  # quit
            break

    cap.release()
    cv.destroyAllWindows()

