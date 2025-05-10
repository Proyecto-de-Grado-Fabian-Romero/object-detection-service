from dataclasses import dataclass

@dataclass
class ViewMetadata:
    filename: str
    yaw: float
    pitch: float
    fov: float
