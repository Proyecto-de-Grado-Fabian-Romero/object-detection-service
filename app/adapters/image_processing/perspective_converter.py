import numpy as np
import cv2
from math import radians, sin, cos
from typing import Tuple


def convert_to_perspective(
    equirect_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    height, width = equirect_img.shape[:2]
    w_out, h_out = output_size

    # Create normalized 3D directions for each pixel
    x = np.linspace(-1, 1, w_out)
    y = np.linspace(-1, 1, h_out)
    xv, yv = np.meshgrid(x, -y)

    zv = np.ones_like(xv)
    directions = np.stack([xv, yv, zv], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    # Apply pitch and yaw (rotations)
    yaw_rad = radians(yaw)
    pitch_rad = radians(pitch)

    Ry = np.array(
        [
            [cos(yaw_rad), 0, sin(yaw_rad)],
            [0, 1, 0],
            [-sin(yaw_rad), 0, cos(yaw_rad)],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, cos(pitch_rad), -sin(pitch_rad)],
            [0, sin(pitch_rad), cos(pitch_rad)],
        ]
    )
    R = Rx @ Ry

    directions_rot = directions @ R.T

    # Convert to spherical coordinates
    lon = np.arctan2(directions_rot[..., 0], directions_rot[..., 2])
    lat = np.arcsin(np.clip(directions_rot[..., 1], -1, 1))

    # Map to equirectangular image coordinates
    u = (lon / (2 * np.pi) + 0.5) * width
    v = (0.5 - lat / np.pi) * height

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    result = cv2.remap(
        equirect_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return result
