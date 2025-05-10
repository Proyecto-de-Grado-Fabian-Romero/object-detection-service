import cv2
import numpy as np
from math import radians, sin, cos
from typing import Tuple

def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    # Convert angles to radians
    yaw = radians(yaw)
    pitch = radians(pitch)

    Ry = np.array([
        [cos(yaw), 0, sin(yaw)],
        [0, 1, 0],
        [-sin(yaw), 0, cos(yaw)],
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, cos(pitch), -sin(pitch)],
        [0, sin(pitch), cos(pitch)],
    ])

    return Rx @ Ry


def convert_to_perspective(
    equirect_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    output_size: Tuple[int, int]
) -> np.ndarray:
    height, width = equirect_img.shape[:2]
    fov_rad = radians(fov)

    w_out, h_out = output_size
    K = np.array([
        [w_out / (2 * np.tan(fov_rad / 2)), 0, w_out / 2],
        [0, h_out / (2 * np.tan(fov_rad / 2)), h_out / 2],
        [0, 0, 1]
    ])

    R = rotation_matrix(yaw, pitch)

    persp_img = cv2.warpPerspective(
        equirect_img,
        K @ R @ np.linalg.inv(K),
        (w_out, h_out),
        flags=cv2.INTER_LINEAR
    )

    return persp_img
