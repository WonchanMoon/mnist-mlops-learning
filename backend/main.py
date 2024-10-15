import numpy as np
from fastapi import FastAPI
from fastapi import BackgroundTasks
from urllib.parse import urlparse

import mlflow
# from mlflow.tracking import MlflowClient
from mlflow.client import MlflowClient
from ml.train import Trainer
from ml.models import LinearModel, CNNModel
from ml.data import load_mnist_data
from ml.utils import set_device
from backend.models import DeleteApiData, TrainApiData, PredictApiData, EvaluateApiData


#mlflow.set_tracking_uri('sqlite:///backend.db')
# mlflow.set_tracking_uri("sqlite:///db/backend.db")
mlflow.set_tracking_uri("http://127.0.0.1:8080")

app = FastAPI()
mlflowclient = MlflowClient(
    mlflow.get_tracking_uri(), mlflow.get_registry_uri())


def train_model_task(model_name: str, hyperparams: dict, epochs: int, model_type: str):
    """Tasks that trains the model. This is supposed to be running in the background
    Since it's a heavy computation it's better to use a stronger task runner like Celery
    For the simplicity I kept it as a fastapi background task"""

    # Setup env
    device = set_device()
    # Set MLflow tracking
    mlflow.set_experiment("MNIST")
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Train
        if model_type == "Linear":
            # Prepare for training
            print("Loading data...")
            train_dataloader, test_dataloader = load_mnist_data()
            print("Training model")
            model = LinearModel(hyperparams).to(device)
        elif model_type == "Conv":
            # Prepare for training
            print("Loading data...")
            train_dataloader, test_dataloader = load_mnist_data(flatten=False)
            print("Training model")
            model = CNNModel(hyperparams).to(device)
        trainer = Trainer(model, device=device)  # Default configs
        history = trainer.train(epochs, train_dataloader, test_dataloader)

        print("Logging results")
        # Log in mlflow
        for metric_name, metric_values in history.items():
            for metric_value in metric_values:
                mlflow.log_metric(metric_name, metric_value)

        # Register model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(f"{tracking_url_type_store=}")

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(
                model, f"{model_type}Model", registered_model_name=model_name, conda_env=mlflow.pytorch.get_default_conda_env())
        else:
            mlflow.pytorch.log_model(
                model, f"{model_type}Model-MNIST", registered_model_name=model_name)
        # Transition to production. We search for the last model with the name and we stage it to production
        mv = mlflowclient.search_model_versions(
            f"name='{model_name}'")[-1]  # Take last model version
        mlflowclient.transition_model_version_stage(
            name=mv.name, version=mv.version, stage="production")
        # mlflowclient.update_model_version(
        #     name=mv.name, version=mv.version, stage="production"
        # )


@app.get("/")
async def read_root():
    return {"Tracking URI": mlflow.get_tracking_uri(),
            "Registry URI": mlflow.get_registry_uri()}


@app.get("/models")
async def get_models_api():
    """Gets a list with model names"""
    # model_list = mlflowclient.list_registered_models()
    model_list = mlflowclient.search_registered_models()
    model_list = [model.name for model in model_list]
    return model_list


@app.post("/train")
async def train_api(data: TrainApiData, background_tasks: BackgroundTasks):
    """Creates a model based on hyperparameters and trains it."""
    hyperparams = data.hyperparams
    epochs = data.epochs
    model_name = data.model_name
    model_type = data.model_type

    background_tasks.add_task(
        train_model_task, model_name, hyperparams, epochs, model_type)

    return {"result": "Training task started"}

@app.post("/evaluate")
async def evaluate_api(data: EvaluateApiData):
    """Predicts on the provided image"""
    model_name = data.model_name
    # Fetch the last model in production
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/Production"
    )
    # 모델의 최신 Production 버전 정보 가져오기
    latest_versions = mlflowclient.get_latest_versions(name=model_name, stages=["Production"])
    latest_version = latest_versions[0]  # 가장 최신 버전 정보 가져오기

    # 모델의 run ID 추출
    run_id = latest_version.run_id

    # run ID로부터 metrics 조회
    run_data = mlflowclient.get_run(run_id).data
    metrics = run_data.metrics

    # 기록된 평가지표 출력
    print("Model metrics recorded during training:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    return {"result": metrics}

@app.post("/predict")
async def predict_api(data: PredictApiData):
    """Predicts on the provided image"""
    img = data.input_image
    model_name = data.model_name
    # Fetch the last model in production
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/Production"
    )
    # Preprocess the image
    # Flatten input, create a batch of one and normalize
    img = np.array(img, dtype=np.float32).flatten()[np.newaxis, ...] / 255
    # Postprocess result
    pred = model.predict(img)
    print(pred)
    res = int(np.argmax(pred[0]))
    return {"result": res, "prob": list(pred.tolist())}


@app.post("/delete")
async def delete_model_api(data: DeleteApiData):
    model_name = data.model_name
    # version = data.model_version
    version = None

    if model_name is None:
        return {"result": "model_name is none"}

    if version is None:
        # Delete all versions
        mlflowclient.delete_registered_model(name=model_name)
        response = {"result": f"Deleted all versions of model {model_name}"}
    elif isinstance(version, list):
        for v in version:
            mlflowclient.delete_model_version(name=model_name, version=v)
        response = {
            "result": f"Deleted versions {version} of model {model_name}"}
    else:
        mlflowclient.delete_model_version(name=model_name, version=version)
        response = {
            "result": f"Deleted version {version} of model {model_name}"}
    return response
