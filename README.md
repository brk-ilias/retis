# RETIS: Recursive Elimination of Trivial Intermediate Splits

Implementation of the RETIS algorithm by Karalic (1992) for regression and classification tasks.

## Overview

RETIS (Recursive Elimination of Trivial Intermediate Splits) is a hybrid machine learning algorithm that combines the interpretability of decision trees with the predictive power of linear models. The algorithm builds a decision tree structure and fits linear regression or logistic regression models in the leaf nodes, potentially improving performance over standard decision trees.

### Key Features

- Implementation for both regression and classification tasks
- Hybrid approach combining tree structure with linear models
- Configurable tree depth, minimum samples, and improvement thresholds
- Compatible with scikit-learn API

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository
2. Navigate to the project directory
3. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
retis/
├── src/
│   └── retis/
│       ├── __init__.py          # Package initialization
│       ├── algorithm.py         # RETIS implementation
│       ├── data.py             # Dataset loading utilities
│       └── evaluation.py       # Evaluation metrics and comparison
├── notebooks/
│   └── evaluation.ipynb        # Comprehensive evaluation notebook
├── tests/
│   └── test_retis.py          # Unit tests
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Usage

### Basic Example - Regression

```python
from retis.algorithm import RETISRegressor
from retis.data import load_california_housing_data

# Load data
X_train, X_test, y_train, y_test, _ = load_california_housing_data()

# Create and train model
model = RETISRegressor(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    use_linear_models=True
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Basic Example - Classification

```python
from retis.algorithm import RETISClassifier
from retis.data import load_breast_cancer_data

# Load data
X_train, X_test, y_train, y_test, _, _ = load_breast_cancer_data()

# Create and train model
model = RETISClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    use_linear_models=True
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Parameters

#### RETISRegressor / RETISClassifier

- `max_depth` (int, default=5): Maximum depth of the tree
- `min_samples_split` (int, default=20): Minimum samples required to split an internal node
- `min_samples_leaf` (int, default=10): Minimum samples required at a leaf node
- `use_linear_models` (bool, default=True): Whether to use linear models in leaves (True for RETIS, False for standard tree)
- `min_improvement` (float, default=0.01): Minimum improvement required to keep a split

## Evaluation

The project includes comprehensive evaluation on two benchmark datasets:

### Regression Task: California Housing Dataset

- 20,640 samples
- 8 features
- Target: Median house value

### Classification Task: Breast Cancer Wisconsin Dataset

- 569 samples
- 30 features
- Binary classification (malignant vs benign)

### Running Evaluations

To run the complete evaluation:

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/evaluation.ipynb
```

2. Run all cells to see:
   - Model training and evaluation
   - Performance comparisons with baseline models
   - Visualization of results
   - Confusion matrices and classification reports

### Baseline Models

**Regression:**
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

**Classification:**
- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression

## Algorithm Details

The RETIS algorithm operates in the following phases:

1. **Tree Construction**: Build a decision tree using recursive partitioning based on feature thresholds
2. **Split Evaluation**: Use MSE (regression) or Gini impurity (classification) to find optimal splits
3. **Leaf Model Fitting**: Fit linear regression or logistic regression models in leaf nodes
4. **Pruning**: Eliminate splits that don't provide sufficient improvement (controlled by `min_improvement`)

### Key Differences from Standard Decision Trees

- **Linear Models in Leaves**: Instead of constant predictions, RETIS fits linear models in each leaf region
- **Improved Predictions**: Can capture local linear trends within leaf nodes
- **Hybrid Approach**: Combines global non-linear structure (tree) with local linear models (leaves)

## Testing

Run unit tests:

```bash
pytest tests/
```

## Results Summary

Based on the evaluation notebook, RETIS demonstrates:

- Competitive performance with ensemble methods on both tasks
- Faster training time compared to Random Forest and Gradient Boosting
- Interpretable tree structure with enhanced prediction capability
- Effective handling of both regression and classification problems

## References

- Karalic, A. (1992). "Employing Linear Regression in Regression Tree Leaves". Proceedings of the European Conference on Artificial Intelligence (ECAI).

## License

This project is intended for educational purposes.

## Author

Implementation created for the RETIS algorithm project, based on Karalic 1992.

## Contributing

This is an academic project. For questions or suggestions, please contact the repository maintainer.
