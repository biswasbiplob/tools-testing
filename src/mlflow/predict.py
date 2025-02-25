import os

import mlflow

model_uri = "runs:/a02e7c5201a144929b12bfd8a0d885a6/sklearn-model"

os.environ["MLFLOW_TRACKING_USERNAME"] = "<username>"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<password>"
mlflow.set_tracking_uri(
    "http://k8s-mlflowpo-mlflowtr-81106c8fd7-8274e91cf06e46f8.elb.eu-west-1.amazonaws.com"
)

# This is the input example logged with the model
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
input_data = pyfunc_model.input_example

# Verify the model with the provided input data using the logged dependencies.
# For more details, refer to:
# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
    env_manager="uv",
)
