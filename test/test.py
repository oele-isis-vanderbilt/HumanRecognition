import pathlib
import os

import torch
from torchvision import transforms as T
from torchreid.reid.utils import featureextractor

cwd = pathlib.path(os.path.abspath(__file__)).parent

extractor = featureextractor(
    model_name='osnet_x1_0',
    model_path='weights/osnet_x1_0_imagenet.pth',
    device='cuda'
)

image_list = [str(cwd/'data'/'image1.png'), str(cwd/'data'/'image2.png')]

features = extractor(image_list)
import pdb; pdb.set_trace()
