"""
1. ETL
2. Train
3. Serve
"""
import mlflow

def run(entrypoint, parameters, use_cache=True):
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)

def workflow():
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        etl_run = run("cifar_data_preprocessing", {})
        train_run = run("cifar_train_mlflow", {}, use_cache=False)

if __name__ == '__main__':
    workflow()