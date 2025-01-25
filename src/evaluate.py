import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

os.environ['MLFFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"


# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


def evaluate(data_path,model_path):

    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y= data["Outcome"]    
    ## load the model from the disk

    mlflow.set_tracking_uri("http://127.0.0.1:5000") 

    model = pickle.load(open(model_path,'rb') ) 

    predictions = model.predict(X)

    accuracy = accuracy_score(y,predictions)

        ## log metrics to MLFLOW

    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")

if __name__=="__main__":
    evaluate(params["data"],params["model"])