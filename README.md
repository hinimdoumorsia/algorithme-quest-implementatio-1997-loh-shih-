# QUEST Decision Tree Implementation

<p align="center">
  <strong>Quick, Unbiased, Efficient Statistical Tree</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Algorithm-QUEST%201997-orange" alt="QUEST Algorithm">
</p>

---

A from-scratch implementation of the **QUEST (Quick, Unbiased, Efficient Statistical Tree)** algorithm for classification, following the original research paper:

> **Loh, W.-Y., & Shih, Y.-S. (1997).** *Split selection methods for classification trees.* Statistica Sinica, 7, 815-840.

## Table of Contents

- [What is QUEST?](#what-is-quest)
- [The Problem with Traditional Trees](#the-problem-with-traditional-trees)
- [How QUEST Solves This](#how-quest-solves-this)
- [Mathematical Foundation](#mathematical-foundation)
  - [Variable Selection (Algorithm 3)](#1-variable-selection-algorithm-3)
  - [Split Point Selection (Algorithm 1)](#2-split-point-selection-algorithm-1--qda)
  - [Categorical Variable Handling (Algorithm 2)](#3-categorical-variable-handling-algorithm-2--crimcoords)
  - [Multiclass Handling (Algorithm 4)](#4-multiclass-handling-algorithm-4--super-class-clustering)
  - [Cost-Complexity Pruning](#5-cost-complexity-pruning)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Performance](#performance)
- [References](#references)

---

## What is QUEST?

QUEST is a decision tree algorithm developed by Loh and Shih (1997) that addresses a fundamental limitation of earlier tree algorithms like CART and C4.5: **selection bias**.

This implementation provides:

- **Unbiased variable selection** via statistical hypothesis tests
- **Multiclass classification** (K >= 2 classes) via super-class clustering
- **Mixed data support** (continuous and categorical features)
- **Cost-complexity pruning** for optimal tree size
- **Quadratic Discriminant Analysis (QDA)** for split point selection

---

## The Problem with Traditional Trees

Traditional decision tree algorithms suffer from **selection bias**:

- **CART** has bias toward variables with many possible split points
- **C4.5** has bias toward variables with many categories

This means that even noise variables with many values may be selected over truly predictive variables with fewer values.

---

## How QUEST Solves This

QUEST separates **variable selection** from **split point selection**:

1. **Variable Selection**: Uses statistical hypothesis tests (ANOVA, Chi-square, Levene) that have approximately equal power across different variable types
2. **Split Point Selection**: Uses Quadratic Discriminant Analysis (QDA) to find optimal split points analytically

This approach achieves **unbiased variable selection** — variables are selected based on their true predictive power, not on the number of possible splits they offer.

---

## Mathematical Foundation

### 1. Variable Selection (Algorithm 3)

At each node $t$ with samples $\mathcal{D}_t$, QUEST selects the best splitting variable using statistical tests:

#### For Continuous Variables: ANOVA F-test

For each continuous variable $X_j$ with $K$ classes, compute the F-statistic:

$$F_j = \frac{\text{Between-group variance}}{\text{Within-group variance}} = \frac{\sum_{k=1}^{K} n_k (\bar{x}_{jk} - \bar{x}_j)^2 / (K-1)}{\sum_{k=1}^{K} \sum_{i \in C_k} (x_{ij} - \bar{x}_{jk})^2 / (n-K)}$$

where:
- $n_k$ = number of samples in class $k$
- $\bar{x}_{jk}$ = mean of variable $j$ in class $k$
- $\bar{x}_j$ = overall mean of variable $j$
- $C_k$ = set of samples in class $k$

Under $H_0$ (no difference between class means): $F_j \sim F(K-1, n-K)$

#### For Categorical Variables: Pearson Chi-square Test

Construct the contingency table and compute:

$$\chi^2_j = \sum_{c=1}^{C} \sum_{k=1}^{K} \frac{(O_{ck} - E_{ck})^2}{E_{ck}}$$

where:
- $O_{ck}$ = observed count of category $c$ and class $k$
- $E_{ck} = \frac{n_{c\cdot} \cdot n_{\cdot k}}{n}$ = expected count under independence

Under $H_0$: $\chi^2_j \sim \chi^2((C-1)(K-1))$

#### Variable Selection Decision

1. Compute p-values for all variables
2. Apply **Bonferroni correction**: $\alpha^* = \alpha / M$ where $M$ = number of variables
3. Select variable with smallest p-value (if $p < \alpha^*$)
4. If no variable is significant, use **Levene's test** for variance differences

---

### 2. Split Point Selection (Algorithm 1 — QDA)

Once a variable $X_j$ is selected, QUEST uses **Quadratic Discriminant Analysis** to find the optimal split point.

#### QDA Discriminant Function

For binary classification (classes 0 and 1), the QDA discriminant function is:

$$\delta_k(x) = -\frac{1}{2} \log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k$$

where:
- $\mu_k$ = mean of class $k$
- $\Sigma_k$ = covariance matrix of class $k$
- $\pi_k$ = prior probability of class $k$

#### Univariate Case (Split Point)

For a single variable, this simplifies to finding the roots of:

$$\delta_0(x) - \delta_1(x) = 0$$

which is a quadratic equation in $x$. The solution gives up to **two split points**.

For unequal variances $(\sigma_0^2 \neq \sigma_1^2)$:

$$\left(\frac{1}{\sigma_1^2} - \frac{1}{\sigma_0^2}\right) x^2 + 2\left(\frac{\mu_0}{\sigma_0^2} - \frac{\mu_1}{\sigma_1^2}\right) x + \left(\frac{\mu_1^2}{\sigma_1^2} - \frac{\mu_0^2}{\sigma_0^2} + \log\frac{\sigma_1^2}{\sigma_0^2}\right) = 0$$

For equal variances (LDA case):

$$x^* = \frac{\mu_0 + \mu_1}{2}$$

---

### 3. Categorical Variable Handling (Algorithm 2 — CRIMCOORDS)

For categorical variables with $C$ categories, QUEST uses **CRIMCOORDS** (Correspondence Analysis Coordinates) to transform categories into a continuous score.

#### Contingency Table Decomposition

Given the $C \times K$ contingency table $\mathbf{N}$:

1. Compute row and column marginals: $\mathbf{r} = \mathbf{N}\mathbf{1}_K$, $\mathbf{c} = \mathbf{N}^T\mathbf{1}_C$
2. Compute the standardized residual matrix:

$$\mathbf{S} = \mathbf{D}_r^{-1/2} \left(\frac{\mathbf{N}}{n} - \frac{\mathbf{r}\mathbf{c}^T}{n^2}\right) \mathbf{D}_c^{-1/2}$$

where $\mathbf{D}_r = \text{diag}(\mathbf{r}/n)$ and $\mathbf{D}_c = \text{diag}(\mathbf{c}/n)$

3. Compute the **SVD**: $\mathbf{S} = \mathbf{U}\mathbf{\Lambda}\mathbf{V}^T$

4. The **CRIMCOORDS score** for category $c$ is:

$$\text{score}_c = \left(\mathbf{D}_r^{-1/2}\mathbf{u}_1\right)_c$$

where $\mathbf{u}_1$ is the first left singular vector.

This transforms each category into a **single numeric value**, enabling the use of standard QDA split finding.

---

### 4. Multiclass Handling (Algorithm 4 — Super-Class Clustering)

For $K > 2$ classes, QUEST reduces the problem to binary classification using **2-means clustering on class centroids**.

#### Algorithm

1. **Compute class centroids**: For each class $k \in \{1, \ldots, K\}$:
   $$\boldsymbol{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} \mathbf{x}_i$$

2. **Apply 2-means clustering** to the $K$ centroids $\{\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K\}$

3. **Partition classes** into two super-classes:
   - Super-class A: $\mathcal{S}_A = \{k : \text{centroid}_k \text{ assigned to cluster 0}\}$
   - Super-class B: $\mathcal{S}_B = \{k : \text{centroid}_k \text{ assigned to cluster 1}\}$

4. **Relabel samples**:
   $$\tilde{y}_i = \begin{cases} 0 & \text{if } y_i \in \mathcal{S}_A \\ 1 & \text{if } y_i \in \mathcal{S}_B \end{cases}$$

5. **Apply binary QDA** on $(\mathbf{X}, \tilde{\mathbf{y}})$

This approach ensures that all $K$ classes contribute to the split decision, and the binary splitting machinery works unchanged.

---

### 5. Cost-Complexity Pruning

QUEST uses **CART-style cost-complexity pruning** to find the optimal tree size.

#### Cost-Complexity Measure

For a subtree $T$, the cost-complexity is:

$$R_\alpha(T) = R(T) + \alpha |T|$$

where:
- $R(T) = \sum_{t \in \tilde{T}} \frac{n_t}{n} r(t)$ = misclassification cost (sum over leaves $\tilde{T}$)
- $r(t) = 1 - \max_k \hat{p}_k(t)$ = misclassification rate at leaf $t$
- $|T|$ = number of leaves
- $\alpha \geq 0$ = complexity parameter

#### Pruning Algorithm

1. **Grow a large tree** $T_{\max}$ (minimal stopping criteria)

2. **For each internal node $t$**, compute the effective alpha:
   $$\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}$$
   
   where $T_t$ is the subtree rooted at $t$

3. **Weakest-link pruning**: Find node with minimum $\alpha_{\text{eff}}$ and prune

4. **Generate sequence** of nested subtrees: $T_{\max} \supseteq T_1 \supseteq T_2 \supseteq \cdots \supseteq \{t_0\}$

5. **Select optimal $\alpha$** via cross-validation or held-out set

---

## Implementation Details

### Project Structure

```
                        
QUEST NOTEBOOK/
├── quest_decision_tree.ipynb      # Main implementation notebook
└── README.md                       # Notebook-specific documentation

```

### Core Classes

#### `Node`
Represents a node in the decision tree:
```python
class Node:
    feature_index: int       # Splitting feature index
    threshold: float         # Split threshold
    crimcoords_map: dict     # Category → score mapping (for categorical)
    left_child: Node         # Left subtree (values ≤ threshold)
    right_child: Node        # Right subtree (values > threshold)
    is_leaf: bool            # Whether this is a leaf
    prediction: int          # Class prediction (for leaves)
    class_counts: dict       # {class: count} at this node
    superclass_groups: tuple # (group_A, group_B) for multiclass splits
```

#### `QuestTree`
The main classifier:
```python
class QuestTree:
    def __init__(self, max_depth=20, min_samples_split=10, 
                 min_samples_leaf=5, ccp_alpha=0.0, feature_types=None):
        ...
    
    def fit(self, X, y):
        """Build the QUEST decision tree from training data."""
        ...
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        ...
```

### Key Functions

| Function | Description |
|----------|-------------|
| `compute_superclass_labels()` | Algorithm 4: 2-means clustering on class centroids |
| `anova_test()` | ANOVA F-test for continuous variables |
| `chi_square_test()` | χ² test for categorical variables |
| `levene_test()` | Levene's test for variance equality |
| `find_qda_split()` | QDA-based split point selection |
| `compute_crimcoords()` | CRIMCOORDS transformation for categorical variables |
| `cost_complexity_prune()` | CART-style cost-complexity pruning |

---

## Usage

### Basic Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=500, n_features=8, n_classes=4,
                           n_informative=6, random_state=42)

# Define feature types
feature_types = ['continuous'] * 8

# Create and train QUEST tree
tree = QuestTree(
    max_depth=5,
    min_samples_split=15,
    feature_types=feature_types,
    ccp_alpha=0.01  # Cost-complexity pruning
)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

# Print tree structure
tree.print_tree()
```

### With Categorical Variables

```python
# Mixed feature types
feature_types = ['continuous', 'continuous', 'categorical', 'categorical']

tree = QuestTree(feature_types=feature_types)
tree.fit(X, y)
```

### Cost-Complexity Pruning

```python
# Train with different alpha values
for alpha in [0.0, 0.01, 0.02, 0.05]:
    tree = QuestTree(ccp_alpha=alpha)
    tree.fit(X_train, y_train)
    
    # Apply pruning
    cost_complexity_prune(tree.root_, alpha, len(X_train))
    
    print(f"α={alpha}: leaves={tree.get_n_leaves()}")
```

---

## Performance

### Test Results on Synthetic Data

| Test Case | Classes | Accuracy | Leaves | Depth |
|-----------|---------|----------|--------|-------|
| Binary Classification | 2 | 88.9% | 10 | 4 |
| Multiclass (K=4) | 4 | 60.0% | 15 | 5 |
| With Pruning (α=0.02) | 2 | 80.0% | 4 | - |

### Pruning Effect

```
α = 0.00: leaves = 11, train_acc = 0.9429, test_acc = 0.8889
α = 0.01: leaves =  4, train_acc = 0.9143, test_acc = 0.8000
α = 0.02: leaves =  4, train_acc = 0.9143, test_acc = 0.8000
α = 0.05: leaves =  4, train_acc = 0.9143, test_acc = 0.8000
α = 0.10: leaves =  2, train_acc = 0.7905, test_acc = 0.7333
```

---

## Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0  # For test utilities
```

Install with:
```bash
pip install numpy scipy scikit-learn
```

---

## References

1. **Loh, W.-Y., & Shih, Y.-S. (1997).** Split selection methods for classification trees. *Statistica Sinica*, 7, 815-840.

2. **Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).** *Classification and Regression Trees*. Wadsworth.

3. **Loh, W.-Y. (2011).** Classification and regression trees. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 1(1), 14-23.

4. **Hothorn, T., Hornik, K., & Zeileis, A. (2006).** Unbiased recursive partitioning: A conditional inference framework. *Journal of Computational and Graphical Statistics*, 15(3), 651-674.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<p align="center">
  Made for the ML community
</p>
