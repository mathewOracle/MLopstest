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

from src.train import train, hyperparameter_tuning

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

def test_hyperparameter_tuning(mock_data):
    """Test the hyperparameter_tuning function."""
    X = mock_data.drop(columns=["Outcome"])
    y = mock_data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    best_model = hyperparameter_tuning(X_train, y_train, param_grid)
    assert isinstance(best_model, GridSearchCV)
    assert isinstance(best_model.best_estimator_, RandomForestClassifier)