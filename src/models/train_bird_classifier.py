import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from bird_spectrogram_classifier import BirdClassifier
from spectrogram_dataset import BirdSpectrogramDataModule

# Configuration parameters
root_dir = "data/split_raw"
num_classes = 7 # Replace with the number of bird species
batch_size = 16
max_epochs = 10

# Initialize the DataModule
data_module = BirdSpectrogramDataModule(root_dir, batch_size=batch_size)

# Initialize the model
model = BirdClassifier(num_classes=num_classes)

# Set up the Weights & Biases logger
wandb_logger = WandbLogger(project="bird-classification")

# Train the model
trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger, log_every_n_steps=1)
trainer.fit(model, data_module)

# Save the trained model
trainer.save_checkpoint("bird_classifier.ckpt")
