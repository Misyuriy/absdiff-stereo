import numpy as np
import cv2 as cv


def draw_text_lines(frame: np.ndarray,
                    lines: list[str],
                    origin: tuple,
                    bottom_origin: bool = False,
                    font_scale: float = 0.6,
                    color: tuple = (255, 255, 255),
                    thickness: int = 1,
                    ):

    line_height = int(font_scale * 32)
    top_left_origin = origin
    if bottom_origin:
        top_left_origin = (origin[0], origin[1] - (line_height * len(lines)))

    lines.reverse()
    for i in range(len(lines)):
        cv.putText(frame, lines[i], top_left_origin,
                   cv.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        if bottom_origin:
            top_left_origin = (top_left_origin[0], top_left_origin[1] - line_height)
        else:
            top_left_origin = (top_left_origin[0], top_left_origin[1] + line_height)


def draw_semitransparent_mask(
        frame: np.ndarray,
        mask: np.ndarray,
        color: tuple = (255, 255, 255),
        alpha: float = 0.2
    ):
    color_layer = np.zeros_like(frame)
    color_layer[:] = color

    mask_area = mask > 0
    frame[mask_area] = cv.addWeighted(frame[mask_area], 1 - alpha, color_layer[mask_area], alpha, 0)
    return frame


def draw_semitransparent_mask2(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 255, 255),
    alpha: float = 0.2
):
    y_idx, x_idx = np.nonzero(mask)
    if len(y_idx) == 0:
        return frame

    color_arr = np.array(color, dtype=np.uint8)
    const_color = (alpha * color_arr).astype(np.uint8)
    alpha_inv = 1 - alpha

    roi = frame[y_idx, x_idx]
    np.multiply(roi, alpha_inv, out=roi, casting='unsafe')
    np.add(roi, const_color, out=roi, casting='unsafe')
    frame[y_idx, x_idx] = roi

    return frame
