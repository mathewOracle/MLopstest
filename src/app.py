import os
import joblib
import pandas as pd
import pickle
import yaml
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

pkl_file_path = os.getenv("PKL_FILE_PATH")
# added a comment
# another comment
# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["predict"]


@app.route('/mlops/ping', methods=['GET'])
def ping():
    # Prepare the response
    response = {
        "status": "Success",
        "server_time": datetime.now()
    }

    return jsonify(response), 200

@app.route('/mlops/predict_diabetes', methods=['POST'])

def predict_diabetes():
    # adding a comment
    # Get the sentence from the POST request
    test_data = request.get_json()

     # Convert test input data to DataFrame
    test_df = pd.DataFrame([test_data])

    
    if pkl_file_path:
        path = pkl_file_path
    else:
        path = params["model"]


    ## load the model from the disk

    model = pickle.load(open(path,'rb') ) 

   # Make prediction
    prediction = model.predict(test_df)
    prediction_proba = model.predict_proba(test_df)

    # Print the prediction results
    outcome = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    confidence = prediction_proba[0][prediction[0]]

    # Prepare the response
    response = {
        "Prediction": outcome,
        "Confidence": confidence,
        "version":"1.1"
    }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
