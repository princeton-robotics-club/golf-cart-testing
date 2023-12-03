import torch
from torchvision import transforms as T
import cv2
import numpy as np
import network
from datasets import Cityscapes
from time import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device configuration
stride = 32 # TODO adjust stride based on image size
# Load the model
model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=stride)
checkpoint = torch.load('best_deeplabv3plus_mobilenet_cityscapes_os16.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# Image transformations
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform the image
original_img1 = cv2.imread('street2.jpg')
original_img1 = cv2.cvtColor(original_img1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#original_img1 = cv2.resize(original_img1, (int(2688/1), int(1520/1)))
original_img1 = cv2.resize(original_img1, (500, 500))
img1 = transform(original_img1).unsqueeze(0).to(device)

original_img2 = cv2.imread('street3.jpg')
original_img2 = cv2.cvtColor(original_img2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#original_img2 = cv2.resize(original_img2, (int(2688/1), int(1520/1)))
original_img2 = cv2.resize(original_img2, (500, 500))
img2 = transform(original_img2).unsqueeze(0).to(device)

# Perform segmentation
with torch.no_grad():
    start_time = time()
    for i in range(10):
        if i & 1:
            outputs = model(img1)
        else:
            outputs = model(img2)

        preds = outputs.max(1)[1].detach().cpu().numpy()
    end_time = time()

print("Total time:", end_time - start_time)
print("Average:", (end_time - start_time)/20)

colorized_preds = Cityscapes.decode_target(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
colorized_preds_bgr = cv2.cvtColor(colorized_preds[0], cv2.COLOR_RGB2BGR)
cv2.imshow('Segmentation Result', cv2.resize(colorized_preds_bgr, (int(colorized_preds_bgr.shape[1]/2), int(colorized_preds_bgr.shape[0]/2))))
#cv2.waitKey(0)

# Create a binary mask for the road class (road has train_id 0 in Cityscapes class)
road_mask = (preds == 0).astype('uint8')
road_mask = np.squeeze(road_mask, axis=0)

# Apply the binary mask to the original image
masked_image = cv2.bitwise_and(original_img1, original_img1, mask=road_mask)
#print(torch.cuda.memory_allocated()/(1024*1024))
cv2.imshow('Masked Image', cv2.resize(masked_image, (int(masked_image.shape[1]/2), int(masked_image.shape[0]/2))))
cv2.waitKey(0)