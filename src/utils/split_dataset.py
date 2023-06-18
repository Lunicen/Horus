import os
from sklearn.model_selection import train_test_split
import shutil


def save_split_files(sets, output_directory):
    for set_name, files in sets.items():
        set_path = os.path.join(output_directory, set_name)
        os.makedirs(set_path, exist_ok=True)

        for file in files:
            file_name = os.path.basename(file)
            bird_class = os.path.basename(os.path.dirname(file))

            class_path = os.path.join(set_path, bird_class)
            os.makedirs(class_path, exist_ok=True)

            shutil.copy(file, os.path.join(class_path, file_name))


def split_dataset(
    source_directory, output_directory, wrong_split=False, train_size=0.7
):
    all_files = []
    for root, _, files in os.walk(source_directory):
        for file in files:
            if file.endswith(".mp3"):
                all_files.append(os.path.join(root, file))

    train, remaining = train_test_split(all_files, train_size=train_size)
    valid, test = train_test_split(remaining, test_size=0.5)
    if wrong_split:
        train = train + valid

    sets = {"train": train, "validation": valid, "test": test}

    save_split_files(sets, output_directory)


split_dataset("data/processed", "data/split_aug")
split_dataset("data/raw", "data/split_raw")
split_dataset("data/processed", "data/split_wrong", wrong_split=True)
