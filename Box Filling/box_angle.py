import numpy as np
import cv2


def calculate_box_angle(approx):
    """
    Calculate the rotation angle of the detected box in the image plane.

    Parameters:
    1. approx: Detected four vertices of the box.

    Returns:
    Angle in degrees between the box's long side and the image's horizontal axis, in range (-90, 90].
    """
    points = np.array(approx, dtype=np.float32).reshape(-1, 2)
    (_, _), (rect_w, rect_h), angle = cv2.minAreaRect(points)

    # minAreaRect angle refers to the "width" side; make it refer to the long side
    if rect_w < rect_h:
        angle -= 90

    if angle <= -90:
        angle += 180
    elif angle > 90:
        angle -= 180

    return angle
