# CS-520-SAGA

A Python-based implementation of MLOps workflows.

Experimental evaluation of MLFlow and a custom pipeline built using different components / libraries (Wandb for tracking experiments, Github Actions for designing the workflow). 

(client-server + pipelines architecture)
![Design](https://ml-ops.org/img/mlops-phasen.jpg)
UI Capabilities offloaded to the frameworks.

Challenges with ML deployment (Focus Points):
* Version control (code/data/model for reproducibility)
* Testing (Model Validation)
* Monitoring performance in production
* Automation (CI/CD for training/serving pipelines)
* Scalability

### Evaluation Criterion:
* Traceability - monitoring and logging the entire ML Lifecycle.
* Data Storage - compatibility with external data sources.
* Model Registry - model lineage, model versioning, stage transitions.
* Visualization - dashboarding capabilities.
* Stability
* Scalability
* Usability
* Maintainability
* Ease of automation

Note: Add a sample workflow for image classification on CIFAR.

## Get Started
* conda create environment
`conda create -n mlflow_demos python=3.7`

* conda activate environment
`conda activate mlflow_demos`

* conda install pytorch
`conda install pytorch torchvision torchaudio -c pytorch`

* conda install pytorch lightning
`conda install -c conda-forge pytorch-lightning`

* conda install mlflow
`conda install -c conda-forge mlflow`

* conda export environment for reproducility
`conda env export --name mlflow_demos > conda.yaml`

`cd mlops`

### Data ETL using PTL and MLFlow Tracking service:

`python3 cifar_data_preprocessing.py --exp-name cifar_test_mlflow --dataset-path cifar10/dataset --datamodule-path cifar10/datamodule --batch-size 128`

### Training using PTL and MLFlow Tracking service:

`python3 cifar_train_mlflow.py --exp-name cifar_test_mlflow --num-epochs 5 --datamodule-path cifar10/datamodule`

mlruns directory will be created inside mlops directory with runs and their metadata
<img width="457" alt="project_structure" src="https://user-images.githubusercontent.com/25073753/205217647-964078d1-2214-49ad-877d-c108b516ad03.png">

## Tracking experiments with MLFLow:
`mlflow ui` or `mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns`, use the latter to use Model Registry.

This command will launch the local tracking server at http://127.0.0.1:5000/

If trained and tracked successfully: Information of different experiments, runs and their metadata, metrics, artifacts can be seen on the dashboard.

<img width="1419" alt="mlflow-ui" src="https://user-images.githubusercontent.com/25073753/205218346-685115d0-4a6b-450a-93d2-4b0867a35816.png">

Training and Validation plots can be found under logged metrics:

<img width="1375" alt="metrics" src="https://user-images.githubusercontent.com/25073753/205218693-2cd48920-38f3-4dc7-9fe8-2be63889987b.png">


### View Registered models, change state of models:
<img width="1437" alt="registry" src="https://user-images.githubusercontent.com/25073753/205757842-f1d7cbbf-96a7-4b86-9662-03b268b75de1.png">

### Testing the registered model (can test any version, or model in any stage):
`python3.7 cifar_inference_mlflow.py --model "cifar" --stage="Staging" --test-datasource="cifar10/datamodule"`

### Run a multistep workflow with MLFlow:
Look at the MLProject file for workflow steps and conda.yaml for environment details.
Execute `mlflow run .` to start a workflow with multiple steps. Data between steps can be shared through artifacts.

