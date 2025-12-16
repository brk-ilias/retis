"""
Simple example demonstrating RETIS usage.
"""

import sys

sys.path.insert(0, "src")

from retis.algorithm import RETISRegressor, RETISClassifier
from retis.data import load_california_housing_data, load_breast_cancer_data
from retis.evaluation import train_and_evaluate


def main():
    print("RETIS Algorithm Example")
    print("=" * 60)

    # Regression Example
    print("\n1. REGRESSION TASK - California Housing")
    print("-" * 60)
    X_train, X_test, y_train, y_test, _ = load_california_housing_data()

    regressor = RETISRegressor(
        max_depth=5, min_samples_split=20, min_samples_leaf=10, use_linear_models=True
    )

    print(f"Training on {X_train.shape[0]} samples...")
    regressor, results = train_and_evaluate(
        regressor,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name="RETIS",
        task="regression",
    )

    print(f"Test RMSE: {results['test_rmse']:.4f}")
    print(f"Test R²: {results['test_r2']:.4f}")
    print(f"Test MAE: {results['test_mae']:.4f}")

    # Classification Example
    print("\n2. CLASSIFICATION TASK - Breast Cancer Wisconsin")
    print("-" * 60)
    X_train, X_test, y_train, y_test, _, _ = load_breast_cancer_data()

    classifier = RETISClassifier(
        max_depth=5, min_samples_split=20, min_samples_leaf=10, use_linear_models=True
    )

    print(f"Training on {X_train.shape[0]} samples...")
    classifier, results = train_and_evaluate(
        classifier,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name="RETIS",
        task="classification",
    )

    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test F1 Score: {results['test_f1']:.4f}")
    print(f"Test Precision: {results['test_precision']:.4f}")
    print(f"Test Recall: {results['test_recall']:.4f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
