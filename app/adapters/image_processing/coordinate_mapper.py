import numpy as np
from math import radians, sin, cos, asin, atan2, pi

def perspective_bbox_to_equirectangular(
    bbox: list,  # [xmin, ymin, xmax, ymax] in view coords
    w_out: int,
    h_out: int,
    yaw: float,
    pitch: float,
    fov: float,
    w_eq: int,
    h_eq: int
) -> list:
    """
    Converts the perspective view bbox to an equirectangular image bbox.

    Returns [xmin_eq, ymin_eq, xmax_eq, ymax_eq] in equirectangular pixels.
    """

    # Auxiliary function: from pixel to 3D vector in perspective camera
    def pixel_to_dir(x, y):
        fov_rad = radians(fov)
        # Normalize pixel to [-1,1]
        nx = (x / w_out) * 2 - 1
        ny = 1 - (y / h_out) * 2

        # Calculate direction in local camera (using fov)
        focal = 1 / np.tan(fov_rad / 2)
        dir_cam = np.array([nx, ny, focal])
        dir_cam /= np.linalg.norm(dir_cam)
        return dir_cam

    # Rotation matrices
    yaw_rad = radians(yaw)
    pitch_rad = radians(pitch)

    Ry = np.array([
        [cos(yaw_rad), 0, sin(yaw_rad)],
        [0, 1, 0],
        [-sin(yaw_rad), 0, cos(yaw_rad)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, cos(pitch_rad), -sin(pitch_rad)],
        [0, sin(pitch_rad), cos(pitch_rad)]
    ])

    R = Rx @ Ry

    # Convert each corner of the bbox to equirectangular coordinates
    corners = [
        (bbox[0], bbox[1]),  # xmin, ymin
        (bbox[2], bbox[1]),  # xmax, ymin
        (bbox[2], bbox[3]),  # xmax, ymax
        (bbox[0], bbox[3]),  # xmin, ymax
    ]

    uv_points = []
    for (x, y) in corners:
        dir_cam = pixel_to_dir(x, y)
        dir_world = R @ dir_cam

        lon = atan2(dir_world[0], dir_world[2])  # -pi .. pi
        lat = asin(np.clip(dir_world[1], -1, 1))  # -pi/2 .. pi/2

        u = (lon / (2 * pi) + 0.5) * w_eq
        v = (0.5 - lat / pi) * h_eq
        uv_points.append((u, v))

    u_coords = [p[0] for p in uv_points]
    v_coords = [p[1] for p in uv_points]

    xmin_eq = max(0, min(u_coords))
    xmax_eq = min(w_eq, max(u_coords))
    ymin_eq = max(0, min(v_coords))
    ymax_eq = min(h_eq, max(v_coords))

    return [xmin_eq, ymin_eq, xmax_eq, ymax_eq]
