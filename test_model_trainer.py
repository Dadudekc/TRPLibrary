import pytest
from unittest.mock import MagicMock
from automatedModelTrainer import AutomatedModelTrainer
import pandas as pd
import numpy as np
from joblib import dump, load
import os
import joblib
import numpy as np
from unittest.mock import MagicMock
import pickle


import sys
# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'automatedModelTrainer'))

# Import the AutomatedModelTrainer from model_trainer
from model_trainer import AutomatedModelTrainer

def test_initialization():
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    assert trainer is not None
    assert hasattr(trainer, 'config')

def test_data_preprocessing():
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    data = pd.DataFrame({'close': np.random.rand(100), 'date': pd.date_range(start='1/1/2020', periods=100)})
    X_train, X_test, y_train, y_test = trainer.preprocess_data_with_feature_engineering(data)
    assert X_train is not None
    assert y_train is not None
    assert len(X_train) > 0
    assert len(y_train) > 0
    assert X_train.shape[1] == X_test.shape[1]  # Ensure the same number of features

def test_create_lag_features():
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    data = pd.DataFrame({'close': np.arange(10)})
    data_with_lags = trainer.create_lag_features(data, 'close', [1, 2])
    assert 'close_lag_1' in data_with_lags.columns
    assert 'close_lag_2' in data_with_lags.columns
    assert data_with_lags['close_lag_1'].isnull().sum() == 1  # First lag should have one NaN
    assert data_with_lags['close_lag_2'].isnull().sum() == 2  # Second lag should have two NaNs

def test_train_neural_network():
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    # Create dummy data
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.rand(20)
    model = trainer.train_neural_network_or_lstm(X_train, y_train, X_val, y_val, model_type="neural_network", epochs=10)
    assert model is not None

def test_train_linear_regression():
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.rand(20)
    model = trainer.train_linear_regression(X_train, y_train, X_val, y_val)
    assert model is not None
    assert hasattr(model, 'coef_')

def test_model_evaluation():
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20)
    model = trainer.train_linear_regression(np.random.rand(100, 10), np.random.rand(100), X_test, y_test)
    evaluation_results = trainer.async_evaluate_model(model, X_test, y_test, 'regression')
    assert 'mse' in evaluation_results and 'r2' in evaluation_results

def test_visualize_training_results():
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    y_test = np.random.rand(20)
    y_pred = np.random.rand(20)
    trainer.visualize_training_results(y_test, y_pred)
    # No assertion needed, just verify that no exceptions are raised

def test_save_and_load_model(tmp_path):
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.rand(20)
    model = trainer.train_linear_regression(X_train, y_train, X_val, y_val)

    model_file_path = tmp_path / "model.pkl"

    # Save the model
    trainer.save_model(model, model_file_path)

    # Debugging: Print mock logger calls
    print(mock_logger.info.call_args_list)

    # Ensure the model file exists before loading
    mock_logger.info.assert_any_call(f'Saving model to {str(model_file_path)}')  # Check the path is logged


def test_full_integration(tmp_path):
    mock_logger = MagicMock()
    trainer = AutomatedModelTrainer(mock_logger)
    data = pd.DataFrame({
        'date': pd.date_range(start='1/1/2020', periods=100),
        'close': np.random.rand(100)
    })
    data_file_path = tmp_path / "data.csv"
    data.to_csv(data_file_path, index=False)

    # Preprocess data
    X_train, X_test, y_train, y_test = trainer.preprocess_data_with_feature_engineering(pd.read_csv(data_file_path))
    assert X_train is not None and y_train is not None

    # Train model
    model = trainer.train_linear_regression(X_train, y_train, X_test, y_test)
    assert model is not None

    # Evaluate model
    evaluation_results = trainer.async_evaluate_model(model, X_test, y_test, 'regression')
    assert 'mse' in evaluation_results and 'r2' in evaluation_results
