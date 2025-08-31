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

        if hist_smoothed[min_idx] > 0:
            current_ratio = min(left_peak, right_peak) / hist_smoothed[min_idx]
        else:
            current_ratio = float("inf")

        if current_ratio > max_ratio:
            max_ratio = current_ratio
            best_threshold = min_idx

    return best_threshold


def ground_exists(lightness: np.ndarray, max_zero_ratio: float = 0.2) -> bool:
    hist = cv.calcHist([lightness], [0], None, [256], [0, 256]).flatten()

    zero_ratio = 1 - (cv.countNonZero(hist) / 256)
    return zero_ratio < max_zero_ratio


def detect_sky(
        frame: np.ndarray,
        min_sky_area: float = 0.2,
        texture_dispersion_window: int = 10,
        variance_threshold: int = 60,
        use_texture: bool = False
):
    h, w = frame.shape[:2]

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lightness = hsv[:, :, 2]

    if not ground_exists(lightness):
        return np.ones_like(lightness), True  # весь кадр это небо

    t_lightness = min_valley_threshold(lightness)
    _, light_mask = cv.threshold(lightness.astype(np.uint8), t_lightness, 255, cv.THRESH_BINARY)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    window_size = texture_dispersion_window

    mean = cv.blur(gray, (window_size, window_size))
    mean_sq = cv.blur(gray.astype(np.float32) ** 2, (window_size, window_size))
    variance = np.maximum(mean_sq - mean ** 2, 0)
    var_norm = cv.normalize(variance, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    t_var_norm = min_valley_threshold(var_norm)
    _, texture_mask = cv.threshold(var_norm, t_var_norm, 255, cv.THRESH_BINARY)

    # объединение масок и убираем шум
    combined = cv.bitwise_and(light_mask, texture_mask)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))

    sky_mask = np.zeros_like(combined)

    # ищем регион неба
    contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:  # возвращаем черную маску - неба в кадре нет
        return sky_mask, False

    min_contour_area = w * h * min_sky_area
    large_contours = [c for c in contours if cv.contourArea(c) > min_contour_area]

    if not large_contours:  # возвращаем черную маску - неба в кадре нет
        return sky_mask, False

    largest_contour = max(large_contours, key=cv.contourArea)
    cv.drawContours(sky_mask, [largest_contour], -1, 255, -1)

    sky_mask = cv.morphologyEx(sky_mask, cv.MORPH_CLOSE, kernel)  # убрать мелкие отверстия

    return sky_mask, False
