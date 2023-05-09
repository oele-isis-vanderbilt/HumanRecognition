import torch
import torchvision.models as models
import torchvision.transforms as transforms

import cv2

from PIL import Image

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Remove the last fully connected layer
model.fc = torch.nn.Sigmoid()

# Define image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the two images
# image1 = Image.open("image1.png")
# image2 = Image.open("image2.png")
image1 = cv2.imread("image1.png", cv2.IMREAD_UNCHANGED)[:,:,:3]
image2 = cv2.imread("image1.png", cv2.IMREAD_UNCHANGED)[:,:,:3]

# image1 = cv2.resize(image1, (244,244))
# image2 = cv2.resize(image2, (244,244))

# Apply the transformations to the images
image1_tensor = transform(image1)
image2_tensor = transform(image2)

# Add an extra dimension to the tensors to represent the batch size of 1
image1_tensor = image1_tensor.unsqueeze(0)
image2_tensor = image2_tensor.unsqueeze(0)

# Pass the images through the model to get the embeddings
with torch.no_grad():
    embedding1 = model(image1_tensor)
    embedding2 = model(image2_tensor)

# Compute the cosine similarity between the two embeddings
cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
print("Cosine similarity between the two images:", cosine_similarity.item())
