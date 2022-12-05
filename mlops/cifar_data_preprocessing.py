import argparse
import os

from cifar10.cifar_datamodule import CIFARDataModule
import mlflow
import torch

MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    

def download_and_transform_data(args):
    experiment = mlflow.get_experiment_by_name(args.exp_name)
    exp_id = experiment.experiment_id if experiment else mlflow.create_experiment(args.exp_name)

    with mlflow.start_run(experiment_id = exp_id) as mlrun:
        cifar10_dm = CIFARDataModule(args)
        isExist = os.path.exists(args.datamodule_path)
        if not isExist:
            print("Creating the datamodule directory..")
            os.makedirs(args.datamodule_path)
        storage_path = os.path.join(args.datamodule_path, "cifar10_dm.pkl")
        torch.save(cifar10_dm, storage_path)
        mlflow.log_artifact(storage_path, args.datamodule_path)
        print("Data extracted, transformed and saved at ", storage_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exp-name")
    parser.add_argument("--dataset-path", default="cifar10/dataset")
    parser.add_argument("--datamodule-path", default="cifar10/datamodule")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(args)
    download_and_transform_data(args)