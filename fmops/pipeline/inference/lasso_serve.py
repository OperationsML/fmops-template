######################################## Start Boilerplate code
import sys
import os

# Vars
MODEL_NAME = os.environ.get("MODEL_NAME")
VERSION = os.environ.get("VERSION")
STAGE = os.environ.get("STAGE")
sys.path.append(os.environ.get("PYTHONPATH"))

# Artifact path
artifact_path = os.path.join(os.environ.get("MODEL_PATH"), f"{VERSION}")

# Load packages once env has been set
import mlflow
import joblib
import numpy as np
import json
import boto3

# API docs
from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel

# Load model
# TODO change model loader to modeltracker load model (add in switch case class to make it easier for DS)
lasso_model = mlflow.sklearn.load_model(artifact_path)

# Load preprocessor
preproccessor = joblib.load(f"{artifact_path}/preprocessor.joblib")
######################################## End Boilerplate code

# FastApi definition
app = FastAPI(title=f"{MODEL_NAME}", version=0.1, root_path=f"/{STAGE}/")

lambda_client = boto3.client("lambda")

class SingleModelData(BaseModel):
    record_id: str
    data: list

class BatchModelData(BaseModel):
    record_ids: list
    data: list

def send_to_monitor_registry(payload):
    return lambda_client.invoke(
        FunctionName=f"model-monitoring-lambda",
        InvocationType="Event",
        LogType="None",
        Payload=json.dumps(payload),
    )

@app.get("/healthcheck")
def healthcheck():
    return f"Lambda is running"

@app.post("/lasso/single")
def single(data: SingleModelData):
    logged = 'no'

    data_array = np.array(data.data)
    x_pred = preproccessor.transform(data_array)
    pred = lasso_model.predict(x_pred)

    # asynch request to store data to database
    if np.random.rand() <= 1:
        payload = {"body": {'record_id': data.record_id, 'prediction': pred[0], 'model_name': MODEL_NAME, 'version': VERSION, 'stage': STAGE}}
        send_to_monitor_registry(data)
        logged = 'yes'

    return {'id': data.record_id, 'prediction': pred.tolist(), 'logged': logged}

@app.post("/lasso/batch")
def batch(data: BatchModelData):

    data_array = np.array(data.data)
    x_pred = preproccessor.transform(data_array)
    pred = lasso_model.predict(x_pred)

    return {'ids': data.record_ids, 'prediction': pred.tolist()}

def lambda_handler(event, context):

    print(event)
    if event.get("source") == "schedule":
        return True

    model_handler = Mangum(app)
    response = model_handler(event, context)
    return response
