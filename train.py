import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
import torch.backends.cudnn as cudnn 

import torchvision 
from torchvision import models
import matplotlib.pyplot as plt
import time 
import os 
from tempfile import TemporaryDirectory

from data_utils import dataloaders, dataset_sizes, class_names, device, imshow

cudnn.benchmark = True
plt.ion()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Keep track of the best model (highest validation accuracy) seen during training,
    # since the final epoch isn't necessarily the best one
    with TemporaryDirectory() as tmpdir:
        best_model_params_path = os.path.join(tmpdir, "best_model_params.pt")
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # Gradients are only needed during training
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {best_acc:.4f}")

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.figure()

    with torch.no_grad():
        for inputs, labels in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)



if __name__ == "__main__":

    # Feature extraction: freeze the pretrained backbone, only train the last layer
    model_conv = torchvision.models.resnet18(weights="IMAGENET1K_V1") 

    for param in model_conv.parameters():
        param.requires_grad = False 

    num_ftrs = model_conv.fc.in_features 
    model_conv.fc = nn.Linear(num_ftrs, len(class_names)) 

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    #Entrainement et évaluation
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25) #on entraine le modèle pendant 25 epochs

    visualize_model(model_conv) 

    # Save permanently so predict.py can reuse these weights without retraining
    torch.save(model_conv.state_dict(), "model_final.pt")
    print("Model saved to model_final.pt")

    plt.ioff()
    plt.show()
