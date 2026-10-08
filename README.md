# Image Classifier with Transfer Learning (PyTorch)

A PyTorch image classifier built with transfer learning (ResNet18). It uses feature extraction: the pretrained network is frozen and only the last layer is trained, which works well on small datasets. I trained it on a tiger vs lion dataset, but it works with any number of custom classes.

This project is based on the official PyTorch tutorial: [Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). The core code (data loading, training loop, prediction) comes from there, split into several files.
