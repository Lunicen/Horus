import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from bird_spectrogram_classifier import BirdClassifier
from BirdClassifierResNet import BirdClassifierResNet
from spectrogram_dataset import BirdSpectrogramDataModule

import wandb

# Configuration parameters
root_dirs = {
    "split1": "data/split_raw",
    "split2": "data/split_aug",
    "split3": "data/split_wrong",
}
num_classes = 7 # Replace with the number of bird species
batch_sizes = [16, 32, 64]  # list of batch sizes you want to run with
max_epochs = 10

def main():
    # Initialize the early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=True, mode='min')

    for batch_size in batch_sizes:
        for split_name, root_dir in root_dirs.items():
            # Initialize the DataModule
            data_module = BirdSpectrogramDataModule(root_dir, batch_size=batch_size)

            # Initialize the model
            model = BirdClassifier(num_classes=num_classes)

            # Set up the Weights & Biases logger
            wandb_logger = WandbLogger(project="bird-classification", name=f"{split_name}_batch_{batch_size}_run")

            # Define checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                filename=f"{split_name}_bird_classifier_batch_{batch_size}",
                monitor='val_loss',  # The metric to monitor
                save_top_k=1,  # Save only the top 1 models based on the metric monitored
                mode='min',  # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
                verbose=True  # Report when a new best model is saved
            )

            # Train the model
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                logger=wandb_logger,
                callbacks=[checkpoint_callback, early_stopping],  # Add callbacks here
                log_every_n_steps=1,
            )
            trainer.fit(model, data_module)

            # Save the trained model (The best model is saved by the checkpoint_callback)
            # trainer.save_checkpoint(f"{split_name}_bird_classifier_batch_{batch_size}.ckpt")
            wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()
