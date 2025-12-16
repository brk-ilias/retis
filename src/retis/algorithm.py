"""
RETIS Algorithm Implementation
Based on Karalic 1992: "Employing Linear Regression in Regression Tree Leaves"

RETIS (Regression Tree Induction System) is a hybrid algorithm
that combines decision tree structure with linear regression models in leaf nodes.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression


class Node:
    """A node in the RETIS tree."""

    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.model = None  # Linear model in leaf nodes
        self.prediction = None  # For classification leaf nodes
        self.n_samples = 0

    def predict(self, X):
        """Predict using the node's model."""
        if self.model is not None:
            return self.model.predict(X)
        return np.full(len(X), self.prediction)


class RETISRegressor(BaseEstimator, RegressorMixin):
    """
    RETIS Regressor: Decision tree with linear regression models in leaves.

    The algorithm works in two phases:
    1. Build a decision tree structure
    2. Replace trivial intermediate splits by fitting linear models in leaves

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree
    min_samples_split : int, default=20
        Minimum samples required to split an internal node
    min_samples_leaf : int, default=10
        Minimum samples required at a leaf node
    use_linear_models : bool, default=True
        Whether to use linear regression in leaves (True for RETIS, False for standard tree)
    min_improvement : float, default=0.01
        Minimum improvement required to keep a split
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        use_linear_models=True,
        min_improvement=0.01,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.use_linear_models = use_linear_models
        self.min_improvement = min_improvement
        self.root = None
        self.n_features_in_ = None

    def _calculate_mse(self, y):
        """Calculate mean squared error."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _find_best_split(self, X, y):
        """Find the best split for the current node."""
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, None

        best_mse = float("inf")
        best_feature = None
        best_threshold = None
        current_mse = self._calculate_mse(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                left_mse = self._calculate_mse(y[left_mask])
                right_mse = self._calculate_mse(y[right_mask])

                weighted_mse = (
                    np.sum(left_mask) * left_mse + np.sum(right_mask) * right_mse
                ) / n_samples

                if weighted_mse < best_mse:
                    improvement = (current_mse - weighted_mse) / current_mse
                    if improvement >= self.min_improvement:
                        best_mse = weighted_mse
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold, best_mse

    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        node = Node()
        node.n_samples = len(y)

        # Check stopping criteria
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            node.is_leaf = True
            self._fit_leaf_model(node, X, y)
            return node

        # Find best split
        feature, threshold, mse = self._find_best_split(X, y)

        if feature is None:
            node.is_leaf = True
            self._fit_leaf_model(node, X, y)
            return node

        # Create split
        node.feature = feature
        node.threshold = threshold

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _fit_leaf_model(self, node, X, y):
        """Fit a linear model or constant prediction in a leaf node."""
        if (
            self.use_linear_models and len(y) >= 3
        ):  # Need at least 3 samples for linear regression
            try:
                model = LinearRegression()
                model.fit(X, y)
                node.model = model
            except:
                # Fallback to mean prediction
                node.prediction = np.mean(y)
        else:
            node.prediction = np.mean(y)

    def _predict_sample(self, x, node):
        """Predict a single sample by traversing the tree."""
        if node.is_leaf:
            if node.model is not None:
                return node.model.predict(x.reshape(1, -1))[0]
            return node.prediction

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def fit(self, X, y):
        """
        Fit the RETIS regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
        """
        X = np.array(X)
        y = np.array(y)
        self.n_features_in_ = X.shape[1]

        self.root = self._build_tree(X, y)
        return self

    def predict(self, X):
        """
        Predict regression targets for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        return predictions


class RETISClassifier(BaseEstimator, ClassifierMixin):
    """
    RETIS Classifier: Decision tree with logistic regression models in leaves.

    Similar to RETISRegressor but adapted for classification tasks.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree
    min_samples_split : int, default=20
        Minimum samples required to split an internal node
    min_samples_leaf : int, default=10
        Minimum samples required at a leaf node
    use_linear_models : bool, default=True
        Whether to use logistic regression in leaves
    min_improvement : float, default=0.01
        Minimum improvement required to keep a split
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        use_linear_models=True,
        min_improvement=0.01,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.use_linear_models = use_linear_models
        self.min_improvement = min_improvement
        self.root = None
        self.classes_ = None
        self.n_features_in_ = None

    def _calculate_gini(self, y):
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _find_best_split(self, X, y):
        """Find the best split for the current node."""
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, None

        best_gini = float("inf")
        best_feature = None
        best_threshold = None
        current_gini = self._calculate_gini(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                left_gini = self._calculate_gini(y[left_mask])
                right_gini = self._calculate_gini(y[right_mask])

                weighted_gini = (
                    np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini
                ) / n_samples

                if weighted_gini < best_gini:
                    improvement = (current_gini - weighted_gini) / (
                        current_gini + 1e-10
                    )
                    if improvement >= self.min_improvement:
                        best_gini = weighted_gini
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def _build_tree(self, X, y, depth=0):
        """Recursively build the tree."""
        node = Node()
        node.n_samples = len(y)

        # Check stopping criteria
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node.is_leaf = True
            self._fit_leaf_model(node, X, y)
            return node

        # Find best split
        feature, threshold, gini = self._find_best_split(X, y)

        if feature is None:
            node.is_leaf = True
            self._fit_leaf_model(node, X, y)
            return node

        # Create split
        node.feature = feature
        node.threshold = threshold

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _fit_leaf_model(self, node, X, y):
        """Fit a logistic model or majority class in a leaf node."""
        if self.use_linear_models and len(y) >= 5 and len(np.unique(y)) > 1:
            try:
                model = LogisticRegression(max_iter=1000, solver="lbfgs")
                model.fit(X, y)
                node.model = model
            except:
                # Fallback to majority class
                node.prediction = np.bincount(y.astype(int)).argmax()
        else:
            node.prediction = np.bincount(y.astype(int)).argmax()

    def _predict_sample(self, x, node):
        """Predict a single sample by traversing the tree."""
        if node.is_leaf:
            if node.model is not None:
                return node.model.predict(x.reshape(1, -1))[0]
            return node.prediction

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def fit(self, X, y):
        """
        Fit the RETIS classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
        """
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.root = self._build_tree(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted class labels
        """
        X = np.array(X)
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X = np.array(X)
        n_samples = len(X)
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            if node.model is not None:
                probabilities[i] = node.model.predict_proba(x.reshape(1, -1))[0]
            else:
                probabilities[i, node.prediction] = 1.0

        return probabilities
