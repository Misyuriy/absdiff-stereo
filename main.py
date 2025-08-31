import numpy as np
import cv2 as cv
from collections import deque
from threading import Thread, Lock
import time
import cProfile

from calibration import calibrate_full, calibrate_continuous, calibrate_sky_full, calibrate_sky_continuous, cropped_mask
from detection import detect_objects
from sky_detection import detect_sky, min_valley_threshold

from opencv_utils import draw_text_lines, draw_semitransparent_mask2


def calibration_worker(frames_to_calibrate, border, max_deviation, use_sky_calibration: bool = False):
    global v_shift, h_shift, calibration_in_progress, calibration_lock, full_calibration_needed, sky_crop_pixels

    while True:
        if frames_to_calibrate:
            calibration_in_progress = True
            frame = frames_to_calibrate.popleft()

            if full_calibration_needed:
                full_calibration_needed = False
                if use_sky_calibration:
                    new_v, new_h = calibrate_sky_full(frame, border, sky_crop_pixels)
                else:
                    new_v, new_h = calibrate_full(frame, border)
            else:
                with calibration_lock:
                    prev_v = v_shift
                    prev_h = h_shift

                if use_sky_calibration:
                    new_v, new_h = calibrate_sky_continuous(frame, border, prev_v, prev_h, max_deviation, sky_crop_pixels)
                else:
                    new_v, new_h = calibrate_continuous(frame, border, prev_v, prev_h, max_deviation)

            with calibration_lock:
                v_shift, h_shift = new_v, new_h

            calibration_in_progress = False
        time.sleep(0.001)  # prevent high CPU usage


def display_histogram(frame, origin: tuple, values: np.ndarray, threshold_value: int):
    hist_height = 100
    hist_width = 256
    hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    hist = cv.calcHist([values], [0], None, [256], [0, 256])
    cv.normalize(hist, hist, 0, hist_height, cv.NORM_MINMAX)

    for i in range(256):
        if hist[i][0] == 0:
            cv.line(hist_img, (i, hist_height), (i, 0), (100, 100, 100), 1)
        else:
            cv.line(hist_img, (i, hist_height), (i, hist_height - int(hist[i][0])), (255, 255, 255), 1)

    cv.line(hist_img, (threshold_value, 0), (threshold_value, hist_height), (0, 0, 255), 1)

    y, x = origin
    frame[x:x + hist_height, y:y + hist_width] = hist_img

    return frame


if __name__ == "__main__":
    file_name = "videos/stereo-0724-04.mp4"

    cv.namedWindow("resized", cv.WINDOW_NORMAL)
    cv.setWindowProperty("resized", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty("resized", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    (x, y, width, height) = cv.getWindowImageRect("resized")

    cap = cv.VideoCapture(file_name)
    ret, frame = cap.read()

    border: int = 30
    sky_crop_pixels: int = 16
    max_deviation: int = 5

    fy = frame.shape[0]
    fx = frame.shape[1]
    f = min((float(width) / (float(fx / 2) - 2 * border)),
            (float(height) / (float(fy) - 2 * border)))
    cv.resizeWindow("resized",
                    int((fx / 2 - 2 * border) * f),
                    int((fy - 2 * border) * f))

    v_shift, h_shift = calibrate_full(frame, border)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))

    show_diff: bool = False

    previous_frame_time: float = time.time()
    frame_count: int = 0

    full_calibration_needed: bool = False
    full_calibration_interval: int = 10  # полная калибровка каждые 10 кадров, чтобы не сдуреть

    calibration_in_progress: bool = False
    calibration_lock = Lock()
    frames_to_calibrate = deque()
    calibration_thread = Thread(target=calibration_worker, args=(frames_to_calibrate, border, max_deviation, True))
    calibration_thread.daemon = True
    calibration_thread.start()

    pr = cProfile.Profile()
    pr.enable()
    #fourcc = cv.VideoWriter_fourcc(*'MP4V')
    #out = cv.VideoWriter('videos/output.mp4', fourcc, 30.0, (1500, 1100))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_count += 1
        if frame_count % full_calibration_interval == 0:
            full_calibration_needed = True

        if (not calibration_in_progress) and (not frames_to_calibrate):
            frames_to_calibrate.append(frame.copy())

        with calibration_lock:
            current_v_shift, current_h_shift = v_shift, h_shift

        right = frame[
                border:fy - border,
                fx // 2 + border:fx - border]
        left = frame[
               border + current_v_shift:fy - border + current_v_shift,
               border + current_h_shift:fx // 2 - border + current_h_shift]

        # уменьшаем разрешение до 800х600
        right_small = cv.resize(right, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        left_small = cv.resize(left, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

        sky_right, fs_right = detect_sky(right_small)
        sky_left, fs_left = detect_sky(left_small)
        sky = cv.bitwise_and(sky_right, sky_left)

        sky_cropped = cropped_mask(sky, sky_crop_pixels // 2)  # обрезать веточки всякие по краям

        # увеличиваем маску неба обратно
        sky_cropped = cv.resize(sky_cropped, (left.shape[1], left.shape[0]), interpolation=cv.INTER_NEAREST)

        right_gray = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
        left_gray = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(right_gray, left_gray)

        diff = cv.bitwise_and(diff, diff, mask=sky_cropped)

        diff_threshold = min_valley_threshold(diff)

        # absdiff маска
        _, diff_mask = cv.threshold(diff, diff_threshold, 255, cv.THRESH_BINARY)
        #diff_mask = cv.morphologyEx(diff_mask, cv.MORPH_OPEN, kernel)  # удаление шума
        diff_mask = cv.morphologyEx(diff_mask, cv.MORPH_CLOSE, kernel)  # заполнение пробелов

        boxes = detect_objects(diff_mask, min_area=5, max_area=100)

        if show_diff:
            display = cv.cvtColor(diff, cv.COLOR_GRAY2BGR)
        else:
            display = left

        #display = draw_semitransparent_mask2(display, sky_cropped, color=(255, 0, 0), alpha=0.3)

        display = display_histogram(display, (1500 - 256, 50), diff, diff_threshold)

        for (x, y, w, h, area) in boxes:
            cv.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text_y = y - 40
            if text_y < 10:  # если объект вверху кадра
                text_y = y + h + 20  # показываем текст под объектом

            cv.putText(display, f"{area} px", (x, text_y),
                       cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        sky_area: float = (cv.countNonZero(sky) * 4.0) / (left.shape[0] * left.shape[1])
        status_text = [
            f"Sky area: {round(sky_area * 100, 2)}%",
            f"Detecting {len(boxes)} objects",
            f"h_shift: {current_h_shift} px",
            f"v_shift: {current_v_shift} px",
        ]

        draw_text_lines(display, status_text, (32, 32), bottom_origin=False, color=(0, 255, 255))

        fps: float = 1 / (time.time() - previous_frame_time)
        previous_frame_time = time.time()
        cv.putText(display, f"{round(fps, 2)} FPS", (display.shape[1] - 128, 32),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)

        cv.imshow("resized", display)
        #out.write(display)

        key = cv.waitKey(1)
        if key == ord("s"):  # show/hide absdiff
            show_diff = not show_diff
        if key == ord("q"):  # quit
            break

    #out.release()
    cap.release()
    cv.destroyAllWindows()

    pr.disable()
    pr.print_stats(sort='cumtime')

