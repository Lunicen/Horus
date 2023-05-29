# (spectrogram_dataset.py file content)
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from AudioProcessor import AudioPreprocessor


class BirdSpectrogramDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split
        self.data = []
        self.label_map = {}
        self.audio_preprocessor = AudioPreprocessor()

        for class_dir in self.root_dir.joinpath(split).glob("*"):
            if class_dir.name not in self.label_map:
                self.label_map[class_dir.name] = len(self.label_map)
            for file in class_dir.glob("*.mp3"):
                if file is None:
                    continue
                trimed_audio, sr = self.audio_preprocessor.load_and_trim_audio(file)
                if trimed_audio is None:
                    continue
                mel_spec = self.audio_preprocessor.create_mel_spectrogram_from_audio(
                    trimed_audio, sr
                )
                label = self.label_map[class_dir.name]
                self.data.append((mel_spec, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spec, label = self.data[idx]
        mel_spec = torch.from_numpy(mel_spec).unsqueeze(0)  # Add channel dimension
        mel_spec = mel_spec.expand(3, -1, -1)  # Expand single channel to three channels
        return mel_spec, label


class BirdSpectrogramDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=16, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = self._create_datasets(
                "train", "validation"
            )
        if stage == "test" or stage is None:
            self.test_dataset, _ = self._create_datasets("test")

    def _create_datasets(self, *splits):
        datasets = []
        for split in splits:
            dataset = BirdSpectrogramDataset(self.root_dir, split=split)
            datasets.append(dataset)
        return datasets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
