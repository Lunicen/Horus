import shutil

numpy_path = "data/numpy"
processed_path = "data/processed"
spectrograms_path = "data/spectrograms"
split_aug_path = "data/split_aug"
split_raw_path = "data/split_raw"
split_wrong_path = "data/split_wrong"
bird_classification_path = "bird-classification"


# Delete the folder and its contents
shutil.rmtree(numpy_path)
shutil.rmtree(processed_path)
shutil.rmtree(spectrograms_path)
shutil.rmtree(split_aug_path)
shutil.rmtree(split_raw_path)
shutil.rmtree(split_wrong_path)
shutil.rmtree(bird_classification_path)
