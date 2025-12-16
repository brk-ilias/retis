"""
Unit tests for RETIS algorithm.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

import sys

sys.path.insert(0, "../src")

from src.retis.algorithm import RETISRegressor, RETISClassifier
from src.retis.data import load_california_housing_data, load_breast_cancer_data
from src.retis.evaluation import evaluate_regression, evaluate_classification


class TestRETISRegressor:
    """Tests for RETISRegressor."""

    def setup_method(self):
        """Set up test data."""
        X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def test_fit_predict(self):
        """Test basic fit and predict."""
        model = RETISRegressor(max_depth=3)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.y_test)
        assert model.n_features_in_ == self.X_train.shape[1]

    def test_with_linear_models(self):
        """Test RETIS with linear models in leaves."""
        model = RETISRegressor(max_depth=3, use_linear_models=True)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.y_test)

    def test_without_linear_models(self):
        """Test RETIS without linear models (standard tree)."""
        model = RETISRegressor(max_depth=3, use_linear_models=False)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.y_test)

    def test_min_samples_constraints(self):
        """Test minimum sample constraints."""
        model = RETISRegressor(max_depth=3, min_samples_split=50, min_samples_leaf=20)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.y_test)

    def test_single_sample_prediction(self):
        """Test prediction on single sample."""
        model = RETISRegressor(max_depth=3)
        model.fit(self.X_train, self.y_train)
        prediction = model.predict(self.X_test[0:1])

        assert len(prediction) == 1


class TestRETISClassifier:
    """Tests for RETISClassifier."""

    def setup_method(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def test_fit_predict(self):
        """Test basic fit and predict."""
        model = RETISClassifier(max_depth=3)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.y_test)
        assert model.n_features_in_ == self.X_train.shape[1]
        assert len(model.classes_) == 2

    def test_predict_proba(self):
        """Test probability predictions."""
        model = RETISClassifier(max_depth=3)
        model.fit(self.X_train, self.y_train)
        probabilities = model.predict_proba(self.X_test)

        assert probabilities.shape == (len(self.y_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_with_linear_models(self):
        """Test RETIS with logistic models in leaves."""
        model = RETISClassifier(max_depth=3, use_linear_models=True)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.y_test)

    def test_without_linear_models(self):
        """Test RETIS without linear models (standard tree)."""
        model = RETISClassifier(max_depth=3, use_linear_models=False)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.y_test)


class TestDataLoaders:
    """Tests for data loading functions."""

    def test_load_california_housing(self):
        """Test California Housing data loader."""
        X_train, X_test, y_train, y_test, feature_names = load_california_housing_data()

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert len(feature_names) == X_train.shape[1]

    def test_load_breast_cancer(self):
        """Test Breast Cancer data loader."""
        X_train, X_test, y_train, y_test, feature_names, target_names = (
            load_breast_cancer_data()
        )

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert len(feature_names) == X_train.shape[1]
        assert len(target_names) == 2


class TestEvaluation:
    """Tests for evaluation functions."""

    def test_evaluate_regression(self):
        """Test regression evaluation."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RETISRegressor(max_depth=3)
        model.fit(X_train, y_train)

        results = evaluate_regression(model, X_train, X_test, y_train, y_test)

        assert "test_rmse" in results
        assert "test_r2" in results
        assert "test_mae" in results
        assert results["test_rmse"] >= 0

    def test_evaluate_classification(self):
        """Test classification evaluation."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RETISClassifier(max_depth=3)
        model.fit(X_train, y_train)

        results = evaluate_classification(model, X_train, X_test, y_train, y_test)

        assert "test_accuracy" in results
        assert "test_f1" in results
        assert "test_precision" in results
        assert "test_recall" in results
        assert 0 <= results["test_accuracy"] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
