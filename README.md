# Image Classifier with Transfer Learning (PyTorch)

A PyTorch image classifier built with transfer learning (ResNet18). It uses feature extraction: the pretrained network is frozen and only the last layer is trained, which works well on small datasets. I trained it on a tiger vs lion dataset, but it works with any number of custom classes.

This project is based on the official PyTorch tutorial: [Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). 

## How it works

1. `split_dataset.py` splits a folder of images into `train/` and `val/` sets (80% / 20%).
2. `train.py` loads ResNet18 pretrained on ImageNet, freezes it, replaces the last layer to match the number of classes, trains it, and saves the best weights to `model_final.pt`.
3. `predict.py` loads `model_final.pt` and predicts the class of any image.
