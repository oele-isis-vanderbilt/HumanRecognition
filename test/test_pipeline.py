# Built-in Imports
from typing import Literal, List
import os
import pathlib
import logging

logger = logging.getLogger(__name__)

# Third-party Imports
import torch
import cv2
import pytest
import numpy as np
import imutils

from mmreid import MMReIDPipeline, render

# CONSTANTS
CWD = pathlib.Path(os.path.abspath(__file__)).parent
TEST_VIDEO = CWD / 'data' / 'PETS09-S2L1.mp4'

assert TEST_VIDEO.exists()

@pytest.fixture
def pipeline():

    if torch.cuda.is_available():
        return MMReIDPipeline(weights=CWD/'weights'/'crowdhuman.pt', device='cuda')
    else:
        return MMReIDPipeline(weights=CWD/'weights'/'crowdhuman.pt', device='cpu')

def test_step_processing(pipeline):

    cap = cv2.VideoCapture(str(TEST_VIDEO), 0)

    # Then perform homography
    while True:

        # Get video
        ret, frame = cap.read()

        if ret:

            # Apply homography
            tracks = pipeline.step(frame)
            output = render(frame, tracks)
            cv2.imshow("output", imutils.resize(output, width=1000))

            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        else:
            break

    # Closing the video
    cv2.destroyAllWindows()
