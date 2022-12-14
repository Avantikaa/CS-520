import pytorch_lightning as pl
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger

import torch
# print("GPUs available: ",torch.cuda.device_count())
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10

import argparse
from cifar10 import utils
from cifar10.module import Module
import warnings
import wandb
import os
os.environ['WANDB_API_KEY'] = '2b5e640eebb24c5d55d40a699278c9b1a24eea8a' #add your account's API key
wandb.login()

warnings.filterwarnings("ignore")

utils.set_seed(0)

        
class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "Sample Validation Set Performance":[wandb.Image(x, caption=f"Predicted:{pred}, True:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })



def main(hparams):
    wandb.init() #initialize wandb run
    cifar_module = Module(hparams)  #get the cifar dataset
    #--------------Wandb functionality------------------   
    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(cifar_module.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    #----------------------end--------------------------
    #W&B provides a lightweight wrapper for logging your ML experiments, used to seamlessly log metrics, model weights, media and more
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train') 
    
    #implementing model checkpointing and early stopping
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='./wandb_checkpoints/', save_top_k=-1, verbose=True, monitor="val_loss_avg", mode="min"
    )
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
    
    trainer = pl.Trainer(
        logger=wandb_logger, callbacks=[early_stop_callback,ImagePredictionLogger(val_samples),checkpoint_callback], max_epochs=hparams.num_epochs, accelerator="cuda"
    )


        
    trainer.fit(cifar_module)  #fit the model
    predictions = trainer.predict(cifar_module, cifar_module.test_dataloader())  #predict the class on test set
    torch.save(trainer, "saved_model")
    model_artifact_name = "model_run"
    art = wandb.Artifact(model_artifact_name, type="model") #Wandb building block for dataset and model versioning - Flexible and lightweight
    art.add_file(local_path="saved_model")
    wandb.log_artifact(art) 
    torch.save(predictions, "wandb_test_results.pkl")
    torch.save(cifar_module.test_dataloader().dataset, "wandb_test_dataset.pkl")
        
    
    data_artifact = wandb.Artifact('wandb_test_dataset.pkl',type='dataset')
    data_artifact.add_file(local_path='wandb_test_dataset.pkl')
    wandb.log_artifact(data_artifact)
    
    results_artifact = wandb.Artifact('wandb_test_results.pkl',type='dataset')
    results_artifact.add_file(local_path='wandb_test_results.pkl')
    wandb.log_artifact(results_artifact)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp-name", default="Default")
    parser.add_argument("--num-epochs", default=6, type=int)
    parser.add_argument("--tags", action=utils.DictPairParser, metavar=utils.DictPairParser.METAVAR)

    parser = Module.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
