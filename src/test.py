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


def evaluate(model_path):

    # Test input data
    test_data = {
        "Pregnancies": 2,
        "Glucose": 138,
        "BloodPressure": 62,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.127,
        "Age": 47
    }

    # Convert test input data to DataFrame
    test_df = pd.DataFrame([test_data])

    ## load the model from the disk

    model = pickle.load(open(model_path,'rb') ) 

   # Make prediction
    prediction = model.predict(test_df)
    prediction_proba = model.predict_proba(test_df)

    # Print the prediction results
    outcome = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    confidence = prediction_proba[0][prediction[0]]

    print(f"Prediction: {outcome}")
    print(f"Confidence: {confidence:.2f}")

if __name__=="__main__":
    evaluate(params["model"])