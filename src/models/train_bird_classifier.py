import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from bird_spectrogram_classifier import BirdClassifier
from spectrogram_dataset import BirdSpectrogramDataModule

import wandb

# Configuration parameters
root_dirs = {
    "split1": "data/split_raw",
    "split2": "data/split_aug",
    "split3": "data/split_wrong",
}
num_classes = 7 # Replace with the number of bird species
batch_size = 16
max_epochs = 10

for split_name, root_dir in root_dirs.items():
    # Initialize the DataModule
    data_module = BirdSpectrogramDataModule(root_dir, batch_size=batch_size)

    # Initialize the model
    model = BirdClassifier(num_classes=num_classes)

    # Set up the Weights & Biases logger
    wandb_logger = WandbLogger(project="bird-classification", name=f"{split_name}_run")

    # Train the model
    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, data_module)

    # Save the trained model
    trainer.save_checkpoint(f"{split_name}_bird_classifier.ckpt")
    wandb_logger.experiment.finish()
