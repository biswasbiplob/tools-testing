import mlflow

model_uri = 'runs:/b0cfd1a5df5c40ef8965f51b7998ec92/sklearn-model'
mlflow.set_tracking_uri('http://127.0.0.1:5000')

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