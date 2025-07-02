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
        top_left_origin = (top_left_origin[0], top_left_origin[1] - line_height)
