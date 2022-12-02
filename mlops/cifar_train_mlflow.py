import argparse
import os
import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import torch
from cifar10 import utils
from cifar10.module import Module

warnings.filterwarnings("ignore")

utils.set_seed(0)


def main(hparams):
    cifar_module = Module(hparams)

    mlf_logger = MLFlowLogger(experiment_name=hparams.exp_name, tracking_uri="./mlruns", tags=args.tags)

    exp = mlf_logger.experiment.get_experiment_by_name(hparams.exp_name)
    artifacts_dir = os.path.join(exp.artifact_location, mlf_logger.run_id, "artifacts")

    checkpoint_callback = ModelCheckpoint(
        dirpath=artifacts_dir, save_top_k=-1, verbose=True, monitor="val_loss_avg", mode="min"
    )

    trainer = Trainer(
        logger=mlf_logger, callbacks=[checkpoint_callback], max_epochs=hparams.num_epochs, accelerator="cpu"
    )
    trainer.fit(cifar_module)
    predictions = trainer.predict(cifar_module, cifar_module.test_dataloader())
    torch.save(predictions, "test_results.pkl")
    torch.save(cifar_module.test_dataloader().dataset, "test_dataset.pkl")
    mlf_logger.experiment.log_artifact(run_id = mlf_logger.run_id, local_path = "test_dataset.pkl")
    mlf_logger.experiment.log_artifact(run_id = mlf_logger.run_id, local_path = "test_results.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--exp-name", default="Default")
    parser.add_argument("--num-epochs", default=128, type=int)
    parser.add_argument("--tags", action=utils.DictPairParser, metavar=utils.DictPairParser.METAVAR)

    parser = Module.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)