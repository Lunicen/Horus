import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.models.resnet import ResNet, ResNet50_Weights
from torchvision.models import resnet50

class BirdClassifierResNet(pl.LightningModule):
    def __init__(self, num_classes):
        super(BirdClassifierResNet, self).__init__()

        # Load pre-trained ResNet50
        self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Replace the last fully connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.feature_extractor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        metrics = {"val_loss": loss, "val_acc": acc}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
