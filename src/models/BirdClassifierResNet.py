import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.models.resnet import ResNet, ResNet50_Weights
from torchvision.models import resnet50
from torchmetrics import Precision, Accuracy, F1Score,  Recall, ConfusionMatrix, AUROC

class BirdClassifierResNet(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super(BirdClassifierResNet, self).__init__()

        # Load pre-trained ResNet50
        self.feature_extractor = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Replace the last fully connected layer
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, num_classes)

        self.learning_rate = learning_rate

        # Define metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.precision_metric = Precision(num_classes=num_classes)
        self.recall = Recall(num_classes=num_classes)
        self.f1 = F1Score(num_classes=num_classes)
        self.auroc = AUROC(num_classes=num_classes)

    def forward(self, x):
        return self.feature_extractor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.train_acc(preds, y)
        self.precision_metric(preds, y)
        self.recall(preds, y)
        self.f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.val_acc(preds, y)
        self.precision_metric(preds, y)
        self.recall(preds, y)
        self.f1(preds, y)
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        self.auroc(probs, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_precision', self.precision, prog_bar=True)
        self.log('val_recall', self.recall, prog_bar=True)
        self.log('val_f1', self.f1, prog_bar=True)
        self.log('val_auroc', self.auroc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
