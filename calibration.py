import numpy as np
import cv2 as cv
from sky_detection import detect_sky


def cropped_mask(mask: np.ndarray, crop_pixels: int):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*crop_pixels+1, 2*crop_pixels+1))
    mask = cv.erode(mask, kernel)
    h, w = mask.shape

    mask[0:crop_pixels, :] = 0
    mask[h - crop_pixels:h, :] = 0
    mask[:, 0:crop_pixels] = 0
    mask[:, w - crop_pixels:w] = 0

    return mask


def calibrate_continuous(frame: np.ndarray,
                         border: int,
                         prev_v_shift: int,
                         prev_h_shift: int,
                         max_deviation: int = 5) -> tuple:
    v_range = (max(-border + 1, prev_v_shift - max_deviation), min(border, prev_v_shift + max_deviation + 1))
    h_range = (max(-border + 1, prev_h_shift - max_deviation), min(border, prev_h_shift + max_deviation + 1))
    return _calibrate(frame, border, v_range, h_range)


def calibrate_full(frame: np.ndarray, border: int) -> tuple:
    v_range = (-border + 1, border)
    h_range = (-border + 1, border)
    return _calibrate_sky(frame, border, v_range, h_range)


def _calibrate(frame: np.ndarray,
               border: int,
               v_range: tuple,
               h_range: tuple) -> tuple:
    height = frame.shape[0]
    width = frame.shape[1]

    w_half = width // 2

    left_gray = cv.cvtColor(frame[:, :w_half], cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(frame[:, w_half:], cv.COLOR_BGR2GRAY)

    v_shift = 0
    h_shift = 0
    min_mean = float('inf')

    for v in range(v_range[0], v_range[1]):
        left_roi = left_gray[
                   border + v:height - border + v,
                   border:w_half - border]

        right_roi = right_gray[
                    border:height - border,
                    border:w_half - border]

        if left_roi.shape != right_roi.shape:
            continue

        diff = cv.absdiff(right_roi, left_roi)
        mean = cv.mean(diff)[0]

        if mean < min_mean:
            min_mean = mean
            v_shift = v

    min_mean = float('inf')

    for h in range(h_range[0], h_range[1]):
        left_roi = left_gray[
                   border + v_shift:height - border + v_shift,
                   border + h:w_half - border + h]

        right_roi = right_gray[
                    border:height - border,
                    border:w_half - border]

        if left_roi.shape != right_roi.shape:
            continue

        diff = cv.absdiff(right_roi, left_roi)
        mean = cv.mean(diff)[0]

        if mean < min_mean:
            min_mean = mean
            h_shift = h

    return v_shift, h_shift


def calibrate_sky_full(frame: np.ndarray, border: int, mask_crop_pixels: int = 0) -> tuple:
    v_range = (-border + 1, border)
    h_range = (-border + 1, border)
    return _calibrate_sky(frame, border, v_range, h_range, mask_crop_pixels)


def calibrate_sky_continuous(frame: np.ndarray,
                             border: int,
                             prev_v_shift: int,
                             prev_h_shift: int,
                             max_deviation: int = 5,
                             mask_crop_pixels: int = 0) -> tuple:
    v_range = (max(-border + 1, prev_v_shift - max_deviation), min(border, prev_v_shift + max_deviation + 1))
    h_range = (max(-border + 1, prev_h_shift - max_deviation), min(border, prev_h_shift + max_deviation + 1))
    return _calibrate_sky(frame, border, v_range, h_range, mask_crop_pixels)


def _calibrate_sky(frame: np.ndarray,
                   border: int,
                   v_range: tuple,
                   h_range: tuple,
                   mask_crop_pixels: int = 0) -> tuple:
    height = frame.shape[0]
    width = frame.shape[1]

    w_half = width // 2

    left = frame[:, :w_half]
    right = frame[:, w_half:]

    right_small = cv.resize(right, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    left_small = cv.resize(left, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    sky_right, fs_right = detect_sky(right_small)
    sky_left, fs_left = detect_sky(left_small)
    if mask_crop_pixels > 0 and not (fs_right and fs_left):
        sky_right = cropped_mask(sky_right, mask_crop_pixels // 2)
        sky_left = cropped_mask(sky_left, mask_crop_pixels // 2)

    sky_right = cv.resize(sky_right, (w_half, height), interpolation=cv.INTER_NEAREST)
    sky_left = cv.resize(sky_left, (w_half, height), interpolation=cv.INTER_NEAREST)

    left_gray = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

    v_shift = 0
    h_shift = 0
    min_mean = float('inf')

    for v in range(v_range[0], v_range[1]):
        left_roi = left_gray[
                   border + v:height - border + v,
                   border:w_half - border]
        sky_left_roi = sky_left[
                       border + v:height - border + v,
                       border:w_half - border]

        right_roi = right_gray[
                    border:height - border,
                    border:w_half - border]
        sky_right_roi = sky_right[
                        border:height - border,
                        border:w_half - border]

        if left_roi.shape != right_roi.shape:
            continue

        common_sky_mask = cv.bitwise_and(sky_left_roi, sky_right_roi)
        diff = cv.absdiff(right_roi, left_roi)
        mean = cv.mean(diff, mask=common_sky_mask)[0]

        if mean < min_mean:
            min_mean = mean
            v_shift = v

    min_mean = float('inf')

    for h in range(h_range[0], h_range[1]):
        left_roi = left_gray[
                   border + v_shift:height - border + v_shift,
                   border + h:w_half - border + h]
        sky_left_roi = sky_left[
                       border + v_shift:height - border + v_shift,
                       border + h:w_half - border + h]

        right_roi = right_gray[
                    border:height - border,
                    border:w_half - border]
        sky_right_roi = sky_right[
                        border:height - border,
                        border:w_half - border]

        if left_roi.shape != right_roi.shape:
            continue

        common_sky_mask = cv.bitwise_and(sky_left_roi, sky_right_roi)
        diff = cv.absdiff(right_roi, left_roi)
        mean = cv.mean(diff, mask=common_sky_mask)[0]

        if mean < min_mean:
            min_mean = mean
            h_shift = h

    return v_shift, h_shift
