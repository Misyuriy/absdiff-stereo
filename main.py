import numpy as np
import cv2 as cv
from calibration import auto_calibrate_stereo
from detection import detect_objects
from opencv_utils import draw_text_lines


if __name__ == "__main__":
    file_name = "C:/Users/andre/PycharmProjects/turret/videos/stereo_birds_04.mp4"

    cap = cv.VideoCapture(file_name)

    cv.namedWindow("resized", cv.WINDOW_NORMAL)
    cv.setWindowProperty("resized", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty("resized", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    (x, y, width, height) = cv.getWindowImageRect("resized")

    roi_top = 640
    border = 50

    ret, frame = cap.read()

    f = min((float(width) / (float(frame.shape[1] / 2) - 2 * border)),
            (float(height) / (float(frame.shape[0]) - 2 * border)))
    cv.resizeWindow("resized", round(float(frame.shape[1] / 2 - 2 * border) * f),
                    round(float(frame.shape[0] - 2 * border) * f))

    v_shift, h_shift = auto_calibrate_stereo(frame, border)

    is_first_resize = True
    paused = False

    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    trackers = cv.legacy.MultiTracker_create()

    detection_mode = False
    detection_frames = 0
    N_DETECTION_FRAMES = 10

    #MIN_TRACKING_CONFIDENCE = 0.4
    #tracker_confidence = {}
    #tracker_counter = 0

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret: break

        f = min((float(width) / (float(frame.shape[1] / 2) - 2 * border)),
                (float(height) / (float(frame.shape[0]) - 2 * border)))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        right = gray[border:frame.shape[0] - border, round(frame.shape[1] / 2) + border:frame.shape[1] - border]
        left = gray[border + v_shift:frame.shape[0] - border + v_shift,
               border + h_shift:round(frame.shape[1] / 2) - border + h_shift]

        diff = cv.absdiff(right, left)

        # detection/tracking
        boxes = []

        if paused:
            pass # не пихать в историю одинаковые бессмысленные кадры на паузе
        elif detection_mode:
            bg_mask = bg_subtractor.apply(left)
            bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_OPEN, kernel)

            boxes = detect_objects(bg_mask, min_area=60)

            detection_frames -= 1
            if detection_frames == 0:
                detection_mode = False

            if len(boxes) > 0:
                trackers = cv.legacy.MultiTracker_create()
                for box in boxes:
                    box_float = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    trackers.add(cv.legacy.TrackerCSRT_create(), left, box_float)

        elif len(trackers.getObjects()) > 0:
            success, boxes = trackers.update(left)
            if success:
                int_boxes = []
                for box in boxes:
                    # это нужно чтобы рисовать на экране (он только целые ест)
                    x, y, w, h = [int(v) for v in box]
                    int_boxes.append((x, y, w, h))

                boxes = int_boxes

            else:
                boxes = []

        # рисуем bounding boxes и пишем информацию об объекте из карты глубины
        display_diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)

        for box in boxes:
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(display_diff, (x, y), (x + w, y + h), (0, 255, 0), 4)

            object_diff = diff[y:y + h, x:x + w]
            median_diff = np.median(object_diff[object_diff > 0])

            cv.putText(display_diff, f"{median_diff:.1f}", (x, y - 20),
                       cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        status_text = ""
        if paused:
            status_text = "Paused"
        elif detection_mode:
            status_text = "Detecting"
        else:
            status_text = f"Tracking {len(boxes)} objects"
        cv.putText(display_diff, status_text, (32, 32),
                   cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

        controls_text = ["D - detect moving objects",
                         "SPACE - pause",
                         "C - recalibrate",
                         "Q - quit"]
        draw_text_lines(display_diff, controls_text, (32, height), bottom_origin=True, color=(0, 255, 255))

        cv.imshow("resized", display_diff)

        # управление с клавиатуры
        key = cv.waitKey(10)
        if key == ord("d") and not paused:  # detect
            detection_mode = True
            detection_frames = N_DETECTION_FRAMES

            # сбрасываем трекеры
            trackers = cv.legacy.MultiTracker_create()

        if key == ord(" "):  # pause/play
            paused = not paused
        if key == ord("c"):  # recalibrate
            v_shift, h_shift = auto_calibrate_stereo(frame, border)
        if key == ord("q"):  # quit
            break

    cap.release()
    cv.destroyAllWindows()
