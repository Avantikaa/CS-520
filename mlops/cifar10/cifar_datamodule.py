import os
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

class CIFARDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.data_dir = hparams.dataset_path
        self.batch_size = hparams.batch_size
        self.transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (1.0,))])
        self.save_hyperparameters(hparams)

    def __dataloader(self, train: bool) -> DataLoader:
        dataset = CIFAR10(root=self.data_dir, train=train, transform=self.transforms, download=True)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=0)
        return dataloader

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)