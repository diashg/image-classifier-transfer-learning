import torch
import numpy as np 
import torchvision 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os 


# Training uses data augmentation; validation only uses standard preprocessing
# Normalization values come from ImageNet, since the pretrained model expects them
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # 224x224 is the input size expected by ResNet18

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "data"

img_dataset = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"] }

dataloaders = {
    x: torch.utils.data.DataLoader(img_dataset[x], batch_size=4, shuffle=True, num_workers=4)
    for x in ["train", "val"] }

dataset_sizes = {x: len(img_dataset[x]) for x in ["train", "val"]}
class_names = img_dataset["train"].classes

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean # reverse normalization for display
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)



if __name__ == "__main__":
    inputs, classes = next(iter(dataloaders["train"]))
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
