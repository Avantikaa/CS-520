# Goal: update the model in staging with the current model in experiment if it performs better
# This is the first stage toward continual development and deployment with ML workflows
# get the run associated with the model in staging
# compare the performance on recorded metrics and accuracy on held-out dataset
# if current better, update Staging.
# Model can be served in production with schedulers and regular SE updates / QA Tests / Sign offs

import mlflow
import time
import torch
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from pytorch_lightning import Trainer
import argparse

BENCHMARK_ACCURACY = 0.9
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def get_experiment_id(experiment_name):
    current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id=current_experiment['experiment_id']
    return experiment_id


# Wait until the model is ready
def wait_until_ready(model_name, model_version):
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)


def get_latest_model(model_name):
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    latest_model_version = max([model_version_info.version for model_version_info in model_version_infos])
    wait_until_ready(model_name, latest_model_version)
    return latest_model_version


def get_metrics_with_run_id(run_id, experiment_id):
    query = "attributes.run_id = '"+run_id+"'"
    run = mlflow.search_runs([experiment_id], filter_string=query)
    if run.empty:
        print("This run is empty ", run_id)
        return
    model_metrics = run["metrics.val_loss"].iloc[0]
    return model_metrics


def load_py_module(model_name, stage):
    model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{stage}")
    return model


def get_model_in_deployment(model_name, stage):
    model_deployment_runid = None
    for mv in client.search_model_versions(model_name):
        if mv.current_stage == stage:
            model_deployment_runid = mv.run_id
            break
    return model_deployment_runid


def clear_mandate_staging_test(test_model, test_datasource):
    cifar_datamodule = torch.load(test_datasource)
    trainer = Trainer(accelerator="cpu")
    result = trainer.validate(test_model, cifar_datamodule.test_dataloader())
    if result[0]["val_acc_avg"] > BENCHMARK_ACCURACY:
        return True
    return False


def deploy_new_model(model_name, version, test_datasource, stage):
    # doubly test the current model on a held out dataset (brownie)
    test_model = load_py_module(model_name, stage)
    if clear_mandate_staging_test(test_model):
        client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage)
        print("Model in {} updated with model version {}".format(stage, version))
    else:
        print("Does not satisfy mandatory checks for deployment, test-accuracy on held-out dataset low")


def update_deployment(model_name, stage, test_datasource, experiment_name):
    model_deployment_runid = get_model_in_deployment(model_name, stage)
    latest_model_version = get_latest_model(model_name)
    current_model = client.get_model_version(name=model_name,
        version=latest_model_version,
    )

    if not model_deployment_runid:
        # check if current model is worth being deployed by validating metrics and testing on a held-out dataset
        print("Testing this model for deployment passes")
        deploy_new_model(model_name, latest_model_version, test_datasource, stage)
    else:
        experiment_id = get_experiment_id(experiment_name)
        best_model_metrics = get_metrics_with_run_id(model_deployment_runid, experiment_id)
        print("Metrics with the model in deployment ", best_model_metrics)
        latest_model_runid = current_model.run_id
        current_model_metrics = get_metrics_with_run_id(latest_model_runid, experiment_id)
        print("Metrics with the latest model ", current_model_metrics)

        if not current_model_metrics:
            print("Metrics not logged with the latest models")
            return

        if best_model_metrics > current_model_metrics:
            # time to replace the model in STAGE, loss should have been lesser
            deploy_new_model(model_name, latest_model_version, test_datasource, stage)
            # can add a logic to notify developers
            # this model can later be served to production through a scheduler (regular Software Engineering CI/CD)
        else:
            print("Existing deployment is better")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-name")
    parser.add_argument("--stage")
    parser.add_argument("--test-datasource")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    
    print("Update Deployment.. ")
    update_deployment(args.model_name, args.stage, args.test_datasource, args.experiment_name)
