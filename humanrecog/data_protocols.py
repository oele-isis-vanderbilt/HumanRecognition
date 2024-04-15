import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Detection:
    id: int
    tlwh: np.ndarray # Nx4
    confidence: float
    cls: int


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
    track: Track
    confidence: float


@dataclass
class PipelineResults:
    person_detections: List[Detection]
    face_detections: List[Detection]
    tracks: List[Track]