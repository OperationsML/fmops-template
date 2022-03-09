## THIS CODE IS STILL BEING UPDATED
import click
import os
import json
import boto3
import numpy as np
import argparse
import requests
from mlflow.tracking import MlflowClient
from flyermlops.tracking.tracking_objects import ModelTracker


# Sagemaker client
ssm_client = boto3.client("ssm")
param = "/dev/MLFlowTrackingUri"
MLFLOW_TRACKING_URI = ssm_client.get_parameter(Name=param)["Parameter"]["Value"]
lambda_client = boto3.client("lambda")
client = MlflowClient(MLFLOW_TRACKING_URI)


# Re-write as flyermlops class
def test_serve(model_name: str = None, run_id: str = None, stage:str = None):

    rest_api = ssm_client.get_parameter(Name=f"/dev/model-rest-api/fmops-template/{model_name}/stage")["Parameter"]["Value"]
    model_tracker = ModelTracker(mlflow_tracking_uri=MLFLOW_TRACKING_URI)
    data = model_tracker.load_input_example(run_id)
   
    # Loop over multiple times to record data in monitoring registry
    for i in range(0,20):
        payload = {"data": data["data"], 'record_id': i}
        resp = requests.post(rest_api, json=payload) 

    client.log_text(run_id, resp.text, "artifacts/serve_test.txt")
    client.log_text(run_id, rest_api, "artifacts/url.txt")
    client.log_text(run_id, str(resp.elapsed.total_seconds()), "artifacts/latency.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Args used for cicd
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--stage", required=True)

    args, _ = parser.parse_known_args()
    test_serve(model_name=args.model_name, run_id=args.run_id, stage=args.stage)