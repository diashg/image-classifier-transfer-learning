import os
import random
import shutil

# directory paths
train = "data/train"
val = "data/val"


def split_train_val(ratio_val=0.2):
    if not os.path.exists(train):
        print(f"Error : '{train}' folder does not exist")
        return  

    print("Starting separation of training and validation datasets")


    for cat in os.listdir(train):
        path_train_cat = os.path.join(train, cat)

        if os.path.isdir(path_train_cat): 
            all_img = []
            
            for f in os.listdir(path_train_cat):
                if os.path.isfile(os.path.join(path_train_cat, f)) and not f.startswith("."):
                    all_img.append(f)

            total_img = len(all_img)

            if total_img == 0:
                print(f"Folder '{cat}' is empty, we skip it")
                continue

            nb = int(total_img * ratio_val)

            print(f"Folder '{cat}' : {total_img} images found. {nb} images (20%) will be moved")

            
            path_val_cat = os.path.join(val, cat)
            os.makedirs(path_val_cat, exist_ok=True)

            # Select randomly the images to move
            img_select= random.sample(all_img, nb)
 
            for name_img in img_select:
                source = os.path.join(path_train_cat, name_img)
                destination = os.path.join(path_val_cat, name_img)
                shutil.move(source, destination) 

    print("Separation completed successfully. Training and validation datasets are ready.")


if __name__ == "__main__":
    split_train_val(ratio_val=0.2)