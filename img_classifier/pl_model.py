import torch
import torch.nn as nn
from torch.optim import AdamW

import pytorch_lightning as pl
from img_classifier.ResNet18 import ResNet18, Resnet34, Resnet50

class ModelModule(pl.LightningModule):
    def __init__(self, model_type):
        super().__init__()
        self.args = []
        assert model_type in ['Resnet18', 'Resnet34', 'Resnet50'], 'model_type MUST be in [Resnet18, Resnet34, Resnet50]'
        if model_type == 'Resnet18':
            self.model = ResNet18()
        elif model_type == 'Resnet34':
            self.model = Resnet34()
        elif model_type == 'Resnet50':
            self.model = Resnet50()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch['image']

        color = batch['color'].squeeze(-1)
        number = batch['number'].squeeze(-1)

        color_predict, number_predict = self.model(x)

        color_loss = nn.CrossEntropyLoss()(color_predict, color)
        number_loss = nn.CrossEntropyLoss()(number_predict, number)
        loss = color_loss + number_loss

        # logging
        metric = {
            'color_loss': color_loss,
            'number_loss': number_loss,
            'loss': loss
        }

        self.log_dict(metric)
        return loss


    def validation_step(self, batch, batch_idx):
        x = batch['image']
        color = batch['color']
        number = batch['number']

        color_predict, number_predict = self.model(x)
        return (color_predict, number_predict, color, number)


    def validation_epoch_end(self, val_epoch_outputs):
        color_predict = torch.cat([x[0] for x in val_epoch_outputs], dim=0)   # [B, 4]
        number_predict = torch.cat([x[1] for x in val_epoch_outputs], dim=0)  # [B, 10]
        color = torch.cat([x[2] for x in val_epoch_outputs], dim=0).squeeze(-1)  # [B]
        number = torch.cat([x[3] for x in val_epoch_outputs], dim=0).squeeze(-1)

        B = number.size(0)

        color_predict = color_predict.argmax(dim=1)
        number_predict = number_predict.argmax(dim=1)

        color_predict = (color_predict == color)
        number_predict = (number_predict == number)

        acc = torch.logical_and(color_predict, number_predict)
        acc = (acc.sum() / B)

        # logging
        metric = {
            'eval_color_acc': (color_predict.sum() / B),
            'eval_number_acc': (number_predict.sum() / B),
            'eval_acc': acc
        }
        self.log_dict(metric)

    def test_step(self, batch, batch_idx):
        x = batch['image']
        color = batch['color']
        number = batch['number']

        color_predict, number_predict = self.model(x)
        return (color_predict, number_predict, color, number)

    def test_epoch_end(self, test_epoch_outputs):
        color_predict = torch.cat([x[0] for x in test_epoch_outputs], dim=0)
        number_predict = torch.cat([x[1] for x in test_epoch_outputs], dim=0)
        color = torch.cat([x[2] for x in test_epoch_outputs], dim=0).squeeze(1)
        number = torch.cat([x[3] for x in test_epoch_outputs], dim=0).squeeze(1)

        B = number.size(0)

        color_predict = color_predict.argmax(dim=1)
        number_predict = number_predict.argmax(dim=1)

        color_predict = (color_predict == color)
        number_predict = (number_predict == number)

        acc = torch.logical_and(color_predict, number_predict)
        acc = (acc.sum() / B)

        # logging
        metric = {
            'test_color_acc': (color_predict.sum() / B),
            'test_number_acc': (number_predict.sum() / B),
            'test_acc': acc
        }
        self.log_dict(metric)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
        return {'optimizer': optimizer}

