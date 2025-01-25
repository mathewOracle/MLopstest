import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from unittest.mock import patch, MagicMock
import pickle
import yaml
import os
import sys
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.train import train

# Mock data for testing
@pytest.fixture
def mock_data():
    """Fixture to create a mock dataset."""
    np.random.seed(42)
    data = pd.DataFrame({
        "Feature1": np.random.rand(100),
        "Feature2": np.random.rand(100),
        "Outcome": np.random.randint(0, 2, 100)
    })
    return data

@pytest.fixture
def mock_params():
    """Fixture to mock params.yaml content."""
    return {
        "train": {
            "data": "mock_data.csv",
            "model": "mock_model.pkl",
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 10
        }
    }

@patch("builtins.open", new_callable=MagicMock)
@patch("src.train.pd.read_csv")
def test_train(mock_read_csv, mock_open, mock_data, mock_params):
    """Test the train function."""
    # Mock reading of CSV data
    mock_read_csv.return_value = mock_data

    # Mock mlflow to avoid actual calls
    with patch("your_script_name.mlflow.start_run") as mock_mlflow:
        # Run the train function
        train(
            mock_params["train"]["data"],
            mock_params["train"]["model"],
            mock_params["train"]["random_state"],
            mock_params["train"]["n_estimators"],
            mock_params["train"]["max_depth"]
        )
        # Assertions
        mock_read_csv.assert_called_once_with(mock_params["train"]["data"])
        mock_open.assert_called_once_with(mock_params["train"]["model"], 'wb')

        # Check if model is saved
        mock_open().write.assert_called()
        assert mock_mlflow.called
