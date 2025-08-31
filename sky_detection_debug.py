import numpy as np
import cv2 as cv


def min_valley_threshold(values: np.ndarray,
                         k: int = 2,
                         smooth_size: int = 5) -> int:
    hist = cv.calcHist([values], [0], None, [256], [0, 256]).flatten()

    hist = hist / hist.sum()

    # сглаживаем гистограмму
    x = np.arange(-(smooth_size // 2), smooth_size // 2 + 1)
    gaussian_kernel = np.exp(-x ** 2 / (2 * (smooth_size / 3) ** 2))
    gaussian_kernel /= gaussian_kernel.sum()
    hist_smoothed = np.convolve(hist, gaussian_kernel, mode='same')

    local_minima = []
    for i in range(1, 255):
        if (hist_smoothed[i] < hist_smoothed[i - 1] and
                hist_smoothed[i] < hist_smoothed[i + 1]):
            local_minima.append(i)

    best_threshold = None
    max_ratio = 0

    for min_idx in local_minima:
        left_peak = hist_smoothed[:min_idx].max()
        right_peak = hist_smoothed[min_idx + 1:].max()

        if left_peak < hist_smoothed[min_idx] * k:
            continue
        if right_peak < hist_smoothed[min_idx] * k:
            continue

        current_ratio = min(left_peak, right_peak) / hist_smoothed[min_idx]

        if current_ratio > max_ratio:
            max_ratio = current_ratio
            best_threshold = min_idx

    return best_threshold


def detect_sky(
        frame: np.ndarray,
        min_sky_area: float = 0.2,  # доля от площади кадра
        texture_sensitivity: float = 3.0,
        texture_dispersion_window: int = 10
    ):

    h, w = frame.shape[:2]

    # небо по цвету (светлые и ненасыщенные тона)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lightness = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    t_lightness = min_valley_threshold(lightness)
    _, light_mask = cv.threshold(lightness.astype(np.uint8), t_lightness, 255, cv.THRESH_BINARY)

    # небо по текстуре (небо гладкое):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    window_size = texture_dispersion_window

    mean = cv.blur(gray, (window_size, window_size))
    mean_sq = cv.blur(gray.astype(np.float32) ** 2, (window_size, window_size))
    variance = np.maximum(mean_sq - mean ** 2, 0)
    var_norm = cv.normalize(variance, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    t_var_norm = min_valley_threshold(var_norm)
    _, texture_mask = cv.threshold(var_norm, t_var_norm, 255, cv.THRESH_BINARY)

    results = {
        "frame": frame,
        "light_mask": light_mask,
        "dispersion_mask": texture_mask,
        "var_norm": var_norm,
        "t_lightness": t_lightness,
        "t_var_norm": t_var_norm
    }

    return results


def display_histogram(frame, origin: tuple, values: np.ndarray, threshold_value: int):
    hist_height = 100
    hist_width = 256
    hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    hist = cv.calcHist([values], [0], None, [256], [0, 256])
    cv.normalize(hist, hist, 0, hist_height, cv.NORM_MINMAX)

    for i in range(256):
        cv.line(hist_img, (i, hist_height), (i, hist_height - int(hist[i][0])), (255, 255, 255), 1)

    cv.line(hist_img, (threshold_value, 0), (threshold_value, hist_height), (0, 0, 255), 1)

    y, x = origin
    frame[x:x + hist_height, y:y + hist_width] = hist_img

    return frame


def draw_all_masks(detection_results: dict):
    result = detection_results["frame"].copy()
    hsv = cv.cvtColor(result, cv.COLOR_BGR2HSV)
    lightness = hsv[:, :, 2]

    masks = [
        ("dispersion_mask", (255, 255, 0)),
        ("light_mask", (0, 0, 255)),
    ]

    for mask_name, color in masks:
        mask = detection_results[mask_name]

        color_layer = np.zeros_like(result)
        color_layer[:] = color

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(result, contours, -1, color, 2)

    result = display_histogram(result, (0, 0), lightness, detection_results["t_lightness"])
    result = display_histogram(result, (0, 216), detection_results["var_norm"], detection_results["t_var_norm"])

    return result


if __name__ == "__main__":
    file_name = "videos/WIN_20250827_17_42_56_Pro.mp4"
    #file_name = "videos/stereo-0724-04.mp4"
    #file_name = "videos/stereo_birds_04.mp4"

    cv.namedWindow("resized", cv.WINDOW_NORMAL)
    cv.setWindowProperty("resized", cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
    cv.setWindowProperty("resized", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    (x, y, width, height) = cv.getWindowImageRect("resized")

    cap = cv.VideoCapture(file_name)
    ret, frame = cap.read()

    border = 50

    f = min((float(width) / (float(frame.shape[1] / 2) - 2 * border)),
            (float(height) / (float(frame.shape[0]) - 2 * border)))
    cv.resizeWindow("resized", round(float(frame.shape[1] / 2 - 2 * border) * f),
                    round(float(frame.shape[0] - 2 * border) * f))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv.flip(frame, 0)

        right = frame[border:frame.shape[0] - border, round(frame.shape[1] / 2) + border:frame.shape[1] - border]
        left = frame[border:frame.shape[0] - border, border:round(frame.shape[1] / 2) - border]

        detection_results = detect_sky(left)
        result = draw_all_masks(detection_results)

        cv.imshow("resized", result)

        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
