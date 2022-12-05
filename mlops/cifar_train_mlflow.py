import argparse
import os
import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import torch
import mlflow
from cifar10 import utils
from cifar10.litmodel import CIFARLitModule
warnings.filterwarnings("ignore")

utils.set_seed(0)

exp_artifacts_dir = './mlruns'
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"

def main(hparams):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(hparams.exp_name)
    exp_id = experiment.experiment_id if experiment else mlflow.create_experiment(hparams.exp_name)

    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        print(f'Run ID: {run_id}')

        mlflow_logger = MLFlowLogger(experiment_name=hparams.exp_name, tracking_uri=MLFLOW_TRACKING_URI, tags=args.tags)
        mlflow_logger._run_id = run_id

        artifacts_dir = os.path.join(exp_artifacts_dir, run_id, "artifacts")

        checkpoint_callback = ModelCheckpoint(
            dirpath=artifacts_dir, save_top_k=-1, verbose=True, monitor="val_loss_avg", mode="min"
        )

        trainer = Trainer(
            logger=mlflow_logger, callbacks=[checkpoint_callback], max_epochs=hparams.num_epochs, accelerator="cpu"
        )

        cifar_module = CIFARLitModule(hparams)
        dm_path = os.path.join(hparams.datamodule_path, "cifar10_dm.pkl")
        cifar_datamodule = torch.load(dm_path)

        trainer.fit(cifar_module, cifar_datamodule)
        mlflow.pytorch.log_model(cifar_module, "model", registered_model_name="cifar")
        
        predictions = trainer.predict(cifar_module, cifar_datamodule.test_dataloader())
        torch.save(predictions, "test_results.pkl")
        torch.save(cifar_datamodule.test_dataloader().dataset, "test_dataset.pkl")

        mlflow_logger.experiment.log_artifact(run_id = mlflow_logger.run_id, local_path = "test_dataset.pkl")
        mlflow_logger.experiment.log_artifact(run_id = mlflow_logger.run_id, local_path = "test_results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp-name", default="Default")
    parser.add_argument("--num-epochs", default=128, type=int)
    parser.add_argument("--tags", action=utils.DictPairParser, metavar=utils.DictPairParser.METAVAR)
    parser.add_argument("--datamodule-path", default="cifar10/datamodule")
    parser = CIFARLitModule.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)