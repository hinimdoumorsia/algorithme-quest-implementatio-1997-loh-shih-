# questtree - QUEST Decision Tree Classifier

A Python implementation of the QUEST (Quick, Unbiased, Efficient Statistical Tree) algorithm for classification.

This implementation faithfully follows the original research paper:

> Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for classification trees. Statistica Sinica, 7, 815-840.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [API Reference](#api-reference)
- [Algorithm Details](#algorithm-details)
- [Module Structure](#module-structure)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [References](#references)

---

## Installation

No installation required. Simply place the `questtree/` directory in your project and import:

```python
from questtree import QuestTreeClassifier
```

---

## Quick Start

### Basic Usage

```python
from questtree import QuestTreeClassifier

# Create classifier (automatic feature type detection)
clf = QuestTreeClassifier()

# Fit to training data
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
accuracy = clf.score(X_test, y_test)
```

### With Pruning

```python
clf = QuestTreeClassifier(ccp_alpha=0.01)
clf.fit(X_train, y_train)
```

### View Tree Structure

```python
from questtree import print_tree, get_tree_summary

print_tree(clf)
summary = get_tree_summary(clf)
```

---

## Features

### Unbiased Variable Selection

QUEST uses statistical hypothesis tests that have approximately equal power across different variable types, eliminating the selection bias present in CART and C4.5.

### Automatic Feature Type Detection

No need to manually specify feature types. The classifier automatically detects:
- Continuous features (numeric with high cardinality)
- Categorical features (strings, low-cardinality integers)
- Binary features (exactly 2 unique values)

```python
clf = QuestTreeClassifier()
clf.fit(X, y)
print(clf.feature_types_)  # ['continuous', 'categorical', 'binary', ...]
```

### Multiclass Support

Handles K >= 2 classes via Super-Class Clustering (Algorithm 4), which partitions classes into two super-classes using 2-means clustering on class centroids.

### Mixed Data Types

Supports datasets with both continuous and categorical features. Categorical variables are transformed using CRIMCOORDS (Algorithm 2).

### Cost-Complexity Pruning

CART-style post-pruning to find optimal tree complexity:

```python
clf = QuestTreeClassifier(ccp_alpha=0.02)  # Higher alpha = more pruning
```

### Scikit-learn Compatible API

Familiar interface without requiring scikit-learn as a dependency:

```python
clf.fit(X, y)
clf.predict(X)
clf.predict_proba(X)
clf.score(X, y)
clf.get_params()
clf.set_params(**params)
```

---

## API Reference

### QuestTreeClassifier

```python
class QuestTreeClassifier:
    def __init__(
        self,
        alpha=0.05,           # Significance level for hypothesis tests
        max_depth=20,         # Maximum tree depth
        min_samples_split=10, # Minimum samples to attempt split
        min_samples_leaf=5,   # Minimum samples per leaf
        ccp_alpha=0.0,        # Cost-complexity pruning parameter
        feature_types=None,   # Manual feature type specification
        random_state=None     # Random seed for reproducibility
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.05 | Significance level for statistical tests. Bonferroni correction is applied internally. |
| `max_depth` | int | 20 | Maximum depth of tree. Acts as safety limit; pruning is primary regularization. |
| `min_samples_split` | int | 10 | Minimum samples required to attempt a split. |
| `min_samples_leaf` | int | 5 | Minimum samples required in each leaf node. |
| `ccp_alpha` | float | 0.0 | Complexity parameter for cost-complexity pruning. Higher values produce smaller trees. |
| `feature_types` | list | None | Manual specification: 'continuous', 'categorical', or 'binary' per feature. If None, auto-detected. |
| `random_state` | int | None | Random seed for reproducibility. |

#### Attributes (after fitting)

| Attribute | Type | Description |
|-----------|------|-------------|
| `root_` | Node | Root node of the fitted tree |
| `n_features_in_` | int | Number of features |
| `classes_` | ndarray | Unique class labels |
| `n_classes_` | int | Number of classes |
| `feature_types_` | list | Detected/specified feature types |
| `pruning_path_` | list | Sequence of (alpha, n_leaves) from pruning |

#### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Build decision tree from training data |
| `predict(X)` | Predict class labels |
| `predict_proba(X)` | Predict class probabilities |
| `score(X, y)` | Return mean accuracy |
| `get_params(deep=True)` | Get parameters |
| `set_params(**params)` | Set parameters |
| `get_n_leaves()` | Return number of leaves |
| `get_depth()` | Return maximum depth |

### Visualization Functions

```python
from questtree import print_tree, get_tree_summary, get_feature_importances

# Print tree structure
print_tree(clf, feature_names=['feat1', 'feat2'])

# Get summary dictionary
summary = get_tree_summary(clf)

# Get feature importances
importances = get_feature_importances(clf)
```

### Utility Functions

```python
from questtree import accuracy_score, confusion_matrix, classification_report

# Accuracy
acc = accuracy_score(y_true, y_pred)

# Confusion matrix
cm, classes = confusion_matrix(y_true, y_pred)

# Full classification report
report = classification_report(y_true, y_pred)
```

---

## Algorithm Details

### Algorithm 1: Split Point Selection (QDA)

Once a variable is selected, QUEST uses Quadratic Discriminant Analysis to find the optimal split point analytically.

For binary classification with classes having means mu_0, mu_1 and variances sigma_0^2, sigma_1^2:

**Equal variances (LDA case):**
```
threshold = (mu_0 + mu_1) / 2
```

**Unequal variances (QDA case):**

Solve the quadratic equation:
```
A*t^2 + B*t + C = 0

where:
A = 0.5 * (1/sigma_0^2 - 1/sigma_1^2)
B = mu_1/sigma_1^2 - mu_0/sigma_0^2
C = 0.5 * (mu_0^2/sigma_0^2 - mu_1^2/sigma_1^2) + log(sigma_1/sigma_0)
```

### Algorithm 2: CRIMCOORDS

For categorical variables with C categories, QUEST uses CRIMCOORDS to transform categories into numeric scores:

1. One-hot encode categories into matrix V (n x C)
2. Center V by subtracting column means
3. Compute SVD: V = U * Sigma * W^T
4. Apply LDA in the reduced space
5. Map LDA direction back to category scores

### Algorithm 3: Variable Selection

QUEST selects variables using statistical hypothesis tests:

**For continuous variables:** ANOVA F-test
```
F = (Between-group variance) / (Within-group variance)
```

**For categorical variables:** Pearson Chi-square test
```
chi^2 = sum((O_ij - E_ij)^2 / E_ij)
```

**Selection procedure:**
1. Compute p-values for all variables
2. Apply Bonferroni correction: alpha* = alpha / M
3. Select variable with smallest p-value (if p < alpha*)
4. If no variable significant, use Levene's test for variance differences

### Algorithm 4: Super-Class Clustering

For K > 2 classes:

1. Compute class centroids using continuous features
2. Apply 2-means clustering to the K centroids
3. Partition classes into two super-classes based on cluster assignment
4. Relabel samples: super-class A -> 0, super-class B -> 1
5. Apply binary QDA/CRIMCOORDS on relabeled data

### Cost-Complexity Pruning

For a subtree T, the cost-complexity measure is:
```
R_alpha(T) = R(T) + alpha * |T|

where:
- R(T) = misclassification rate
- |T| = number of leaves
- alpha = complexity parameter
```

Effective alpha for node t:
```
alpha_eff(t) = (R(t) - R(T_t)) / (|T_t| - 1)
```

Nodes with smallest alpha_eff are pruned first.

---

## Module Structure

```
questtree/
|-- __init__.py           # Main exports
|-- classifier.py         # QuestTreeClassifier class
|-- node.py               # Node data structure
|-- splits.py             # QDA split & CRIMCOORDS (Algorithms 1 & 2)
|-- variable_selection.py # Statistical tests (Algorithm 3)
|-- superclass.py         # Super-class clustering (Algorithm 4)
|-- pruning.py            # Cost-complexity pruning
|-- feature_detection.py  # Automatic feature type detection
|-- validation.py         # Input validation
|-- plotting.py           # Visualization utilities
|-- utils.py              # Helper functions & metrics
|-- examples.py           # Usage examples
|-- README.md             # This file
```

---

## Examples

### Binary Classification

```python
import numpy as np
from questtree import QuestTreeClassifier

# Generate data
np.random.seed(42)
X = np.random.randn(200, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train
clf = QuestTreeClassifier()
clf.fit(X[:160], y[:160])

# Evaluate
print(f"Accuracy: {clf.score(X[160:], y[160:]):.4f}")
```

### Multiclass Classification

```python
from questtree import QuestTreeClassifier, print_tree

# 4-class problem
X = np.vstack([np.random.randn(50, 2) + center 
               for center in [[0,0], [0,3], [3,0], [3,3]]])
y = np.repeat([0, 1, 2, 3], 50)

clf = QuestTreeClassifier(ccp_alpha=0.01)
clf.fit(X, y)

print(f"Classes: {list(clf.classes_)}")
print(f"Accuracy: {clf.score(X, y):.4f}")
print_tree(clf)
```

### Categorical Features

```python
# Mixed continuous and categorical
X = np.column_stack([
    np.random.randn(200),                          # continuous
    np.random.choice(['A', 'B', 'C'], 200)         # categorical
])
y = np.array([0]*100 + [1]*100)

clf = QuestTreeClassifier()
clf.fit(X, y)

print(f"Feature types: {clf.feature_types_}")
# Output: ['continuous', 'categorical']
```

### Manual Feature Type Specification

```python
clf = QuestTreeClassifier(
    feature_types=['continuous', 'continuous', 'categorical', 'binary']
)
clf.fit(X, y)
```

### Finding Optimal Alpha via Cross-Validation

```python
from questtree import QuestTreeClassifier
import numpy as np

def cross_validate_alpha(X, y, alphas, n_folds=5):
    results = []
    n = len(y)
    fold_size = n // n_folds
    
    for alpha in alphas:
        scores = []
        for i in range(n_folds):
            test_idx = range(i * fold_size, (i + 1) * fold_size)
            train_idx = [j for j in range(n) if j not in test_idx]
            
            clf = QuestTreeClassifier(ccp_alpha=alpha)
            clf.fit(X[train_idx], y[train_idx])
            scores.append(clf.score(X[list(test_idx)], y[list(test_idx)]))
        
        results.append((alpha, np.mean(scores), np.std(scores)))
    
    return results

# Usage
alphas = [0.0, 0.005, 0.01, 0.02, 0.05]
results = cross_validate_alpha(X, y, alphas)

for alpha, mean, std in results:
    print(f"alpha={alpha:.3f}: {mean:.4f} (+/- {std:.4f})")
```

---

## Dependencies

**Required:**
- numpy >= 1.20.0
- scipy >= 1.7.0

**Optional:**
- pandas (for DataFrame input support)
- matplotlib (for plot_pruning_path)

**Not required:**
- scikit-learn

---

## References

1. Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for classification trees. Statistica Sinica, 7, 815-840.

2. Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and Regression Trees. Wadsworth.

3. Loh, W.-Y. (2011). Classification and regression trees. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 1(1), 14-23.

4. Hothorn, T., Hornik, K., & Zeileis, A. (2006). Unbiased recursive partitioning: A conditional inference framework. Journal of Computational and Graphical Statistics, 15(3), 651-674.

---

## License

MIT License

---

## Comparison with Other Trees

| Feature | QUEST | CART | C4.5 |
|---------|-------|------|------|
| Variable Selection | Statistical tests | Exhaustive search | Information gain |
| Selection Bias | Unbiased | Biased to many splits | Biased to many categories |
| Split Finding | QDA (analytical) | Exhaustive search | Exhaustive search |
| Categorical Handling | CRIMCOORDS | Subset search | One-vs-all |
| Pruning | Cost-complexity | Cost-complexity | Pessimistic error |
