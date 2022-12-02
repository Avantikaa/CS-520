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
`conda create -n 520env python=3.10`

* conda install pytorch
`conda install pytorch torchvision torchaudio -c pytorch`

* conda install pytorch lightning
`conda install -c conda-forge pytorch-lightning`

* conda install mlflow
`conda install -c conda-forge mlflow`

Training using PTL and MLFlow Tracking service:

`python3 mlops/cifar_train_mlflow.py --exp-name cifar_test_mlflow --num-epochs 5`

mlruns directory will be created inside mlops directory with runs and their metadata
<img width="457" alt="project_structure" src="https://user-images.githubusercontent.com/25073753/205217647-964078d1-2214-49ad-877d-c108b516ad03.png">
