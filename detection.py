import numpy as np
import cv2 as cv
from calibration import auto_calibrate_stereo
from opencv_utils import draw_text_lines


def detect_objects(mask, min_area=100, min_white_ratio=0.2):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        area = cv.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv.boundingRect(contour)

        roi = mask[y:y + h, x:x + w]
        white_pixels = cv.countNonZero(roi)
        white_ratio = white_pixels / (w * h)

        if white_ratio < min_white_ratio:
            continue

        boxes.append((x, y, w, h))

    return boxes


if __name__ == "__main__":
    file_name = "your/video/path"

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

    fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    while cap.isOpened():
        if not paused:
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

        # detection/tracking
        boxes = []
        if not paused: # не пихать в историю одинаковые бессмысленные кадры на паузе
            fgmask = fgbg.apply(left)
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

            boxes = detect_objects(fgmask, min_area=50)

        # рисуем bounding boxes и пишем информацию об объекте из карты глубины
        display_diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)

        for (x, y, w, h) in boxes:
            cv.rectangle(display_diff, (x, y), (x + w, y + h), (0, 255, 0), 4)

            object_diff = diff[y:y + h, x:x + w]
            median_diff = np.median(object_diff[object_diff > 0])

            cv.putText(display_diff, f"{median_diff:.1f}", (x, y - 20),
                       cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        status_text = ""
        if paused:
            status_text = "Paused"
        else:
            status_text = f"{len(boxes)} objects detected"
        cv.putText(display_diff, status_text, (32, 32),
                   cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

        controls_text = ["SPACE - pause",
                         "C - recalibrate",
                         "Q - quit"]

        draw_text_lines(display_diff, controls_text, (32, height - 32), bottom_origin=True, color=(0, 255, 255))

        cv.imshow("resized", display_diff)

        key = cv.waitKey(10)

        if key == ord(" "):  # pause/play
            paused = not paused
        if key == ord("c"):  # recalibrate
            v_shift, h_shift = auto_calibrate_stereo(frame, border)
        if key == ord("q"):  # quit
            break

    cap.release()
    cv.destroyAllWindows()
