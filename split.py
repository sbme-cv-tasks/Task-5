import os
import shutil
import random

dataset_path = "dataset"
output_path = "split_dataset"

train_ratio = 0.8

train_path = os.path.join(output_path, "train")
test_path = os.path.join(output_path, "test")

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

random.seed(42)

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # create folders
    os.makedirs(os.path.join(train_path, person), exist_ok=True)
    os.makedirs(os.path.join(test_path, person), exist_ok=True)

    # copy train
    for img in train_images:
        src = os.path.join(person_path, img)
        dst = os.path.join(train_path, person, img)
        shutil.copy(src, dst)

    # copy test
    for img in test_images:
        src = os.path.join(person_path, img)
        dst = os.path.join(test_path, person, img)
        shutil.copy(src, dst)

print("Done splitting dataset!")