import argparse
import numpy as np
import os
import torch
from pytorch_lightning import Trainer
import mlflow

MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def predict(parser_args):
    model_name = parser_args.model_name
    stage = parser_args.stage
    model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{stage}")
    dm_path = os.path.join(parser_args.test_datasource, "cifar10_dm.pkl")
    cifar_datamodule = torch.load(dm_path)
    trainer = Trainer(accelerator="cpu")
    result = trainer.predict(model, cifar_datamodule.test_dataloader())
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-name")
    parser.add_argument("--stage")
    parser.add_argument("--test-datasource")
    args = parser.parse_args()
    preds = predict(args)
    print("Predictions.. ",  preds)