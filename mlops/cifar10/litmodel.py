

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as M
from pytorch_lightning import LightningModule


class CIFARLitModule(LightningModule):
    NUM_CLASSES = 10

    def __init__(self, hparams: argparse.Namespace):
        super(CIFARLitModule, self).__init__()

        self.save_hyperparameters(hparams)

        self.__build_model()
        self.__build_criterion()

    def __build_model(self):
        m = self.hparams.model_name

        if m == "resnet50":
            self._model = M.resnet50(pretrained=True)
            self._model.fc = nn.Linear(2048, self.NUM_CLASSES)

        elif m == "vgg16":
            self._model = M.vgg16(pretrained=True)
            self._model.classifier[6] = nn.Linear(4096, self.NUM_CLASSES)

        else:
            raise ValueError(f"Model {m} is not supported.")

    def __build_criterion(self):
        self._criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self._model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)

        loss = self._criterion(outputs, targets)
        self.log("train_loss", loss)
        return {
            "loss": loss,  # required
            "log": {"train_loss": loss},  # for TensorBoard
        }

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)

        loss = self._criterion(outputs, targets)
        acc = torch.sum((targets == torch.argmax(outputs, dim=1))).float() / len(targets)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        loss_avg = torch.stack([x["val_loss"] for x in outputs]).mean()
        acc_avg = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("val_loss_avg", loss_avg.item())
        self.log("val_acc_avg", acc_avg.item())
        return {
            "val_loss_avg": loss_avg.item(),
            "val_acc_avg": acc_avg.item(),
            "log": {"val_loss_avg": loss_avg.item(), "val_acc_avg": acc_avg.item()},  # for TensorBoard
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, targets = batch
        return self(images)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        return [optimizer], [scheduler]
        
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument("--model-name", default="vgg16")
        parser.add_argument("--lr", default=0.001)
        parser.add_argument("--momentum", default=0.8)
        return parser