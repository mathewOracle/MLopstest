import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from src.evaluate import evaluate
import warnings
warnings.filterwarnings("ignore", category=Warning)


@pytest.fixture
def mock_data():
    """Fixture to provide mock dataset."""
    return pd.DataFrame({
        "Feature1": [0.1, 0.2, 0.3, 0.4],
        "Feature2": [1.1, 1.2, 1.3, 1.4],
        "Outcome": [0, 1, 0, 1]
    })

@pytest.fixture
def mock_model():
    """Fixture to provide a mock model."""
    class MockModel:
        def predict(self, X):
            return [0, 1, 0, 1]  # Simulated predictions matching `Outcome`
    return MockModel()

@patch("src.evaluate.pd.read_csv")
@patch("src.evaluate.pickle.load")
@patch("src.evaluate.mlflow.log_metric")
@patch("builtins.open", new_callable=mock_open)
def test_evaluate(mock_open_file, mock_log_metric, mock_pickle_load, mock_read_csv, mock_data, mock_model):
    """Test the evaluate function."""
    # Mock `pd.read_csv` to return the mock dataset
    mock_read_csv.return_value = mock_data

    # Mock `pickle.load` to return the mock model
    mock_pickle_load.return_value = mock_model

    # Call the evaluate function
    evaluate("mock_data_path.csv", "mock_model_path.pkl")

    # Assertions
    mock_read_csv.assert_called_once_with("mock_data_path.csv")
    mock_open_file.assert_called_once_with("mock_model_path.pkl", 'rb')
    mock_pickle_load.assert_called_once()
    mock_log_metric.assert_called_once_with("accuracy", 1.0)  # Accuracy matches mock data and predictions

    print("Test passed with logged accuracy of 1.0.")
