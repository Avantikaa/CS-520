import argparse
import os

from cifar10.cifar_datamodule import CIFARDataModule
import mlflow
import torch
mlflow.set_tracking_uri("sqlite:///mlruns.db")

def download_and_transform_data(args):
    with mlflow.start_run() as mlrun:
        cifar10_dm = CIFARDataModule(args)
        mlflow.log_params(args)
        torch.save(cifar10_dm, "cifar10_dm.pkl")
        mlflow.log_artifact("cifar10_dm.pkl", args.datamodule_path)
        print("Data extracted, transformed and saved at ", args.datamodule_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--dataset-path", default="cifar10/dataset")
    parser.add_argument("--datamodule-path", default="cifar10/datamodule")
    parser.add_argument("--batch-size", default=8)
    args = parser.parse_args()
    print(args)
    download_and_transform_data(args)
    