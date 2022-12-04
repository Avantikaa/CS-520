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

registry_uri = 'sqlite:///mlflow.db'
tracking_uri = 'http://127.0.0.1:5000'

mlflow.tracking.set_registry_uri(registry_uri)
mlflow.tracking.set_tracking_uri(tracking_uri)

def main(hparams):
    with mlflow.start_run() as mlrun:
        cifar_module = CIFARLitModule(hparams)

        mlf_logger = MLFlowLogger(experiment_name=hparams.exp_name, tracking_uri=tracking_uri, tags=args.tags)

        exp = mlf_logger.experiment.get_experiment_by_name(hparams.exp_name)
        artifacts_dir = os.path.join(exp.artifact_location, mlf_logger.run_id, "artifacts")

        checkpoint_callback = ModelCheckpoint(
            dirpath=artifacts_dir, save_top_k=-1, verbose=True, monitor="val_loss_avg", mode="min"
        )

        trainer = Trainer(
            logger=mlf_logger, callbacks=[checkpoint_callback], max_epochs=hparams.num_epochs, accelerator="cpu"
        )
        # Auto log all MLflow entities
        # mlflow.pytorch.autolog()
        mlflow.pytorch.log_model(cifar_module, "model", registered_model_name="cifar")
        cifar_datamodule = torch.load(hparams.datamodule_path)
        trainer.fit(cifar_module, cifar_datamodule)
        predictions = trainer.predict(cifar_module, cifar_datamodule.test_dataloader())
        torch.save(predictions, "test_results.pkl")
        torch.save(cifar_datamodule.test_dataloader().dataset, "test_dataset.pkl")
        mlf_logger.experiment.log_artifact(run_id = mlf_logger.run_id, local_path = "test_dataset.pkl")
        mlf_logger.experiment.log_artifact(run_id = mlf_logger.run_id, local_path = "test_results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp-name", default="Default")
    parser.add_argument("--num-epochs", default=128, type=int)
    parser.add_argument("--tags", action=utils.DictPairParser, metavar=utils.DictPairParser.METAVAR)
    parser.add_argument("--datamodule-path", default="cifar10/datamodule")
    parser = CIFARLitModule.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)