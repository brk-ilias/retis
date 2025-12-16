"""
Dataset loading and preprocessing utilities for RETIS evaluation.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_california_housing_data(test_size=0.2, random_state=42, scale=True):
    """
    Load California Housing dataset for regression tasks.

    Parameters
    ----------
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    scale : bool, default=True
        Whether to standardize features

    Returns
    -------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training targets
    y_test : array-like
        Test targets
    feature_names : list
        Names of features
    """
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data.feature_names


def load_breast_cancer_data(test_size=0.2, random_state=42, scale=True):
    """
    Load Breast Cancer Wisconsin dataset for classification tasks.

    Parameters
    ----------
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    scale : bool, default=True
        Whether to standardize features

    Returns
    -------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training targets
    y_test : array-like
        Test targets
    feature_names : list
        Names of features
    target_names : list
        Names of target classes
    """
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data.feature_names, data.target_names


def get_dataset_info(dataset_name):
    """
    Get information about a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('california_housing' or 'breast_cancer')

    Returns
    -------
    info : dict
        Dictionary containing dataset information
    """
    if dataset_name == "california_housing":
        data = fetch_california_housing()
        return {
            "name": "California Housing",
            "n_samples": data.data.shape[0],
            "n_features": data.data.shape[1],
            "task": "regression",
            "target_description": "Median house value (in $100,000)",
            "description": data.DESCR,
        }
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        return {
            "name": "Breast Cancer Wisconsin",
            "n_samples": data.data.shape[0],
            "n_features": data.data.shape[1],
            "task": "classification",
            "n_classes": len(data.target_names),
            "class_names": data.target_names,
            "description": data.DESCR,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
