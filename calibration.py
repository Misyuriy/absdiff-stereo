import numpy as np
import cv2 as cv
from opencv_utils import draw_text_lines


def auto_calibrate_stereo(frame: np.ndarray, border: int) -> tuple:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    v_shift = 0
    h_shift = 0
    min_mean = float('inf')

    for v in range(-border + 1, border):
        right_roi = gray[border:h - border, w // 2 + border:w - border]

        left_roi = gray[border + v:h - border + v, border:w // 2 - border]

        if left_roi.shape != right_roi.shape:
            continue

        diff = cv.absdiff(right_roi, left_roi)
        current_mean = cv.mean(diff)[0]

        if current_mean < min_mean:
            min_mean = current_mean
            v_shift = v

    min_mean = float('inf')

    for h in range(-border + 1, border):
        right_roi = gray[border:h - border, w // 2 + border:w - border]
        left_roi = gray[border + v_shift:h - border + v_shift, border + h:w // 2 - border + h]

        if left_roi.shape != right_roi.shape:
            continue

        diff = cv.absdiff(right_roi, left_roi)
        current_mean = cv.mean(diff)[0]

        if current_mean < min_mean:
            min_mean = current_mean
            h_shift = h

    return v_shift, h_shift


if __name__ == "__main__":
    file_name = "your/video/path"

    roi_top = 640
    border = 50
    v_shift = 0
    h_shift = 0
    manual_v_shift = 0
    manual_h_shift = 0

    cap = cv.VideoCapture(file_name)

    cv.namedWindow('resized', cv.WINDOW_NORMAL)
    cv.setWindowProperty('resized', cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty('resized', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    (x, y, width, height) = cv.getWindowImageRect('resized')

    ret, frame = cap.read()
    f = min((float(width) / (float(frame.shape[1] / 2) - 2 * border)),
            (float(height) / (float(frame.shape[0]) - 2 * border)))
    cv.resizeWindow('resized', round(float(frame.shape[1] / 2 - 2 * border) * f),
                    round(float(frame.shape[0] - 2 * border) * f))

    paused = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret: break

        f = min((float(width) / (float(frame.shape[1] / 2) - 2 * border)),
                (float(height) / (float(frame.shape[0]) - 2 * border)))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        right = gray[border:frame.shape[0] - border, round(frame.shape[1] / 2) + border:frame.shape[1] - border]
        left = gray[border + v_shift + manual_v_shift:frame.shape[0] - border + v_shift + manual_v_shift,
               border + h_shift + manual_h_shift:round(frame.shape[1] / 2) - border + h_shift + manual_h_shift]

        diff = cv.absdiff(right, left)
        display_diff = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)

        if paused:
            cv.putText(display_diff, "Paused", (32, 32),
                       cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

        controls_text = ["WASD - manual shift",
                         "SPACE - pause",
                         "C - auto calibrate",
                         "Q - quit"]
        draw_text_lines(display_diff, controls_text, (32, height), bottom_origin=True, color=(0, 255, 255))

        cv.imshow('resized', display_diff)

        key = cv.waitKey(10)
        if key == ord('w'):  # UP
            manual_v_shift = manual_v_shift + 1
        if key == ord('s'):  # DOWN
            manual_v_shift = manual_v_shift - 1
        if key == ord('a'):  # LEFT
            manual_h_shift = manual_h_shift - 1
        if key == ord('d'):  # RIGHT
            manual_h_shift = manual_h_shift + 1
        if key == ord(' '):  # pause/play
            paused = not paused

        if key == ord('c'):
            manual_v_shift = 0
            manual_h_shift = 0
            v_shift, h_shift = auto_calibrate_stereo(frame, border)

        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
