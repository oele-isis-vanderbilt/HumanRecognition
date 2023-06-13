from mmreid import Detector

def detector(weights, device):
    return Detector(weights, device=device)

def test_detecting():
    ...
