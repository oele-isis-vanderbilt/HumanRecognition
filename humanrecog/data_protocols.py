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
    bbox: np.ndarray
    confidence: float
    embedding: np.ndarray = None


@dataclass
class ReIDResult:
    bbox: np.ndarray # Nx4
    scores: np.ndarray # Nx1
    ids: np.ndarray # Nx1


@dataclass
class PipelineResults:
    person_detections: List[Detection]
    face_detections: List[Detection]
    # tracks: List[Track]