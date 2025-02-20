import os

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

os.environ["MLFLOW_TRACKING_USERNAME"] = "<username>"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<password>"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.eu-west-1.amazonaws.com"

mlflow.set_tracking_uri(
    "http://k8s-mlflowpo-mlflowtr-81106c8fd7-8274e91cf06e46f8.elb.eu-west-1.amazonaws.com"
)

# Create or get existing experiment with S3 artifact location
experiment_name = "demo-basic-setup"
try:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location=f"s3://airflow-holidu-dev/mlflow/{experiment_name}",
    )
except mlflow.exceptions.MlflowException:
    # If experiment already exists, get its ID
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Set the experiment
mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    params = {"max_depth": 2, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Infer the model signature
    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    # Log parameters and metrics using the MLflow APIs
    mlflow.log_params(params)
    mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-random-forest-reg-model",
    )
