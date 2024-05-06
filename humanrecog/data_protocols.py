import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Detection:
    id: int
    tlwh: np.ndarray # Nx4
    confidence: float
    cls: int
    keypoints: Optional[np.ndarray] = None


@dataclass
class Track:
    frame_id: int
    id: int
    tlwh: np.ndarray
    confidence: float
    embedding: np.ndarray = None

    face: Detection = None
    face_embedding: np.ndarray = None


@dataclass
class ReIDTrack:
    reid: int
    name: str
    track: Track
    cosine: float


@dataclass
class PipelineResults:
    person_detections: List[Detection]
    face_detections: List[Detection]
    tracks: List[Track]
    reid_tracks: List[ReIDTrack]