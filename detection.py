import numpy as np
import cv2 as cv
from calibration import calibrate_full
from opencv_utils import draw_text_lines

import time


def detect_objects(mask, min_area=100, max_area=1000):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        area = cv.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv.boundingRect(contour)
        boxes.append((x, y, w, h, area))

    return boxes


if __name__ == "__main__":
    file_name = "videos/stereo-0724-13.mp4"

    cap = cv.VideoCapture(file_name)

    cv.namedWindow("resized", cv.WINDOW_NORMAL)
    cv.setWindowProperty("resized", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty("resized", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    (x, y, width, height) = cv.getWindowImageRect("resized")

    border = 50

    ret, frame = cap.read()

    f = min((float(width) / (float(frame.shape[1] / 2) - 2 * border)),
            (float(height) / (float(frame.shape[0]) - 2 * border)))
    cv.resizeWindow("resized", round(float(frame.shape[1] / 2 - 2 * border) * f),
                    round(float(frame.shape[0] - 2 * border) * f))

    v_shift, h_shift = calibrate_full(frame, border)

    is_first_resize = True

    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    diff_threshold = 30

    show_mask: bool = False  # показываем маску или же сам кадр
    detection_method: int = 0
    # 0 - absdiff mask & background mask
    # 1 - absdiff mask only
    # 2 - background mask only
    previous_frame_time: float = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        f = min((float(width) / (float(frame.shape[1] / 2) - 2 * border)),
                (float(height) / (float(frame.shape[0]) - 2 * border)))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        right = gray[border:frame.shape[0] - border, round(frame.shape[1] / 2) + border:frame.shape[1] - border]
        left = gray[border + v_shift:frame.shape[0] - border + v_shift,
               border + h_shift:round(frame.shape[1] / 2) - border + h_shift]

        diff = cv.absdiff(right, left)

        # absdiff маска
        _, diff_mask = cv.threshold(diff, diff_threshold, 255, cv.THRESH_BINARY)  # порог (30 дефолт)
        diff_mask = cv.morphologyEx(diff_mask, cv.MORPH_OPEN, kernel)  # удаление шума
        diff_mask = cv.morphologyEx(diff_mask, cv.MORPH_CLOSE, kernel)  # заполнение пробелов

        # background маска
        bg_mask = bg_subtractor.apply(left)
        bg_mask = cv.morphologyEx(bg_mask, cv.MORPH_OPEN, kernel)

        # объединяем
        combined_mask = cv.bitwise_and(bg_mask, diff_mask)
        combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)

        # выбираем маску в зависимости от режима
        masks = [combined_mask, diff_mask, bg_mask]
        used_mask = masks[detection_method]
        boxes = detect_objects(used_mask)

        display = cv.cvtColor(used_mask if show_mask else diff, cv.COLOR_GRAY2BGR)

        for (x, y, w, h, area) in boxes:
            cv.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_y = y - 40
            if text_y < 10:  # если объект вверху кадра
                text_y = y + h + 20  # показываем текст под объектом

            cv.putText(display, f"{area} px", (x, text_y),
                       cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        match detection_method:
            case 0:
                status_text = f"COMBINED mask, detecting {len(boxes)} targets"
            case 1:
                status_text = f"ABSDIFF mask, detecting {len(boxes)} targets"
            case 2:
                status_text = f"BACKGROUND mask, detecting {len(boxes)} targets"

        cv.putText(display, status_text, (32, 32),
                   cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

        controls_text = [
            "1 - use combined mask",
            "2 - use absdiff mask",
            "3 - use background mask",
            "S - show/hide mask",
            "C - recalibrate",
            "Q - quit"
        ]

        draw_text_lines(display, controls_text, (32, height + 32), bottom_origin=True, color=(0, 255, 255))

        fps: float = 1 / (time.time() - previous_frame_time)
        previous_frame_time = time.time()
        cv.putText(display, f"{round(fps, 2)} FPS", (display.shape[1] - 128, 32),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        cv.imshow("resized", display)

        key = cv.waitKey(1)

        if key == ord("1"):
            detection_method = 0
        if key == ord("2"):
            detection_method = 1
        if key == ord("3"):
            detection_method = 2
        if key == ord("s"):  # show/hide mask
            show_mask = not show_mask
        if key == ord("c"):  # recalibrate
            v_shift, h_shift = calibrate_full(frame, border)
        if key == ord("q"):  # quit
            break

    cap.release()
    cv.destroyAllWindows()
