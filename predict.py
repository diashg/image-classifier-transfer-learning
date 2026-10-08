import sys

import torch
import torch.nn as nn 

from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image

from data_utils import data_transforms, class_names, device, imshow

# Rebuild the same architecture used in train.py, then load the trained weights
model = models.resnet18() 
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model_final.pt", weights_only=True))
model = model.to(device)


def visualize_model_predictions(model, img_path):
    was_training = model.training 
    model.eval()

    img = Image.open(img_path)
    img = data_transforms["val"](img) # must match the preprocessing used at training time
    img = img.unsqueeze(0)            # model expects a batch, not a single image
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2, 2, 1)
        ax.axis("off")
        ax.set_title(f"predicted: {class_names[preds[0]]}")
        imshow(img.cpu().data[0])

        model.train(mode=was_training)



if __name__ == "__main__":

    img_path = sys.argv[1]
    visualize_model_predictions(model, img_path)
    plt.ioff()
    plt.show()
    