from pathlib import Path
import pytorch_lightning as pl
from BirdClassifierResNet import BirdClassifierResNet
from bird_spectrogram_classifier import BirdClassifier
from spectrogram_dataset import BirdSpectrogramDataModule
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

root_dirs = {
    "split1": "data/split_raw",
    "split2": "data/split_aug",
    "split3": "data/split_wrong",
}
num_classes = 7 
batch_size = 32  # single batch size
learning_rate = 0.001  # single learning rate

models = [ 
    Path(f"split{num}_bird_classifier_rest.ckpt") for num in range(1,4)
]


def main():
    for split_name, root_dir in root_dirs.items():
        data_module = BirdSpectrogramDataModule(root_dir, batch_size=batch_size)
        data_module.setup()
        data_loders = [("train_subset", data_module.train_dataloader()), ("val_subset", data_module.val_dataloader()), ("test_subset", data_module.test_dataloader())]
        for model_path in models:
            # Initialize the model
            model = BirdClassifierResNet.load_from_checkpoint(model_path, num_classes=num_classes, learning_rate=learning_rate)

            for subset_name, data_loder in data_loders:
                # Set up the Weights & Biases logger
                wandb_logger = WandbLogger(
                    project="bird-classification",
                    name=f"{split_name}_{model_path.name}_{subset_name}",
                )

                # Test the model
                trainer = pl.Trainer(logger=wandb_logger)
                trainer.test(model, dataloaders=data_loder)

                wandb_logger.experiment.config.update({
                        "split_name": split_name,
                        "model_name": model_path.name,
                        "subset_name": subset_name
                    })
                wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()
