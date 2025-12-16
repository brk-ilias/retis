"""
Evaluation framework for RETIS algorithm.
"""

import numpy as np
import time
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd


def evaluate_regression(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Evaluate a regression model.

    Parameters
    ----------
    model : estimator
        Fitted regression model
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training targets
    y_test : array-like
        Test targets
    model_name : str, default="Model"
        Name of the model for display

    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics
    """
    # Training predictions
    start_time = time.time()
    y_train_pred = model.predict(X_train)
    train_time = time.time() - start_time

    # Test predictions
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    test_time = time.time() - start_time

    # Calculate metrics
    results = {
        "model": model_name,
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_time": train_time,
        "test_time": test_time,
    }

    return results


def evaluate_classification(
    model, X_train, X_test, y_train, y_test, model_name="Model"
):
    """
    Evaluate a classification model.

    Parameters
    ----------
    model : estimator
        Fitted classification model
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training targets
    y_test : array-like
        Test targets
    model_name : str, default="Model"
        Name of the model for display

    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics
    """
    # Training predictions
    start_time = time.time()
    y_train_pred = model.predict(X_train)
    train_time = time.time() - start_time

    # Test predictions
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    test_time = time.time() - start_time

    # Calculate metrics
    results = {
        "model": model_name,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred, average="weighted"),
        "test_precision": precision_score(y_test, y_test_pred, average="weighted"),
        "train_recall": recall_score(y_train, y_train_pred, average="weighted"),
        "test_recall": recall_score(y_test, y_test_pred, average="weighted"),
        "train_f1": f1_score(y_train, y_train_pred, average="weighted"),
        "test_f1": f1_score(y_test, y_test_pred, average="weighted"),
        "train_time": train_time,
        "test_time": test_time,
        "confusion_matrix": confusion_matrix(y_test, y_test_pred),
        "classification_report": classification_report(y_test, y_test_pred),
    }

    return results


def compare_models(models, X_train, X_test, y_train, y_test, task="regression"):
    """
    Compare multiple models on the same dataset.

    Parameters
    ----------
    models : dict
        Dictionary of {model_name: fitted_model}
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training targets
    y_test : array-like
        Test targets
    task : str, default='regression'
        Type of task ('regression' or 'classification')

    Returns
    -------
    comparison : pd.DataFrame
        DataFrame containing comparison results
    """
    results = []

    for model_name, model in models.items():
        if task == "regression":
            result = evaluate_regression(
                model, X_train, X_test, y_train, y_test, model_name
            )
        else:
            result = evaluate_classification(
                model, X_train, X_test, y_train, y_test, model_name
            )
        results.append(result)

    # Create DataFrame, excluding non-scalar values
    if task == "classification":
        # Remove confusion matrix and classification report from DataFrame
        df_results = []
        for r in results:
            df_r = {
                k: v
                for k, v in r.items()
                if k not in ["confusion_matrix", "classification_report"]
            }
            df_results.append(df_r)
        comparison = pd.DataFrame(df_results)
    else:
        comparison = pd.DataFrame(results)

    return comparison


def train_and_evaluate(
    model, X_train, X_test, y_train, y_test, model_name="Model", task="regression"
):
    """
    Train and evaluate a model.

    Parameters
    ----------
    model : estimator
        Unfitted model
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training targets
    y_test : array-like
        Test targets
    model_name : str, default="Model"
        Name of the model
    task : str, default='regression'
        Type of task ('regression' or 'classification')

    Returns
    -------
    model : estimator
        Fitted model
    results : dict
        Evaluation results
    """
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    # Evaluate
    if task == "regression":
        results = evaluate_regression(
            model, X_train, X_test, y_train, y_test, model_name
        )
    else:
        results = evaluate_classification(
            model, X_train, X_test, y_train, y_test, model_name
        )

    results["fit_time"] = fit_time

    return model, results
