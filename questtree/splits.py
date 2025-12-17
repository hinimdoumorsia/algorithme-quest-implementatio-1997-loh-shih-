"""
Split Point Selection for QUEST Decision Tree.

Implements:
- Algorithm 1: QDA-based split point selection
- Algorithm 2: CRIMCOORDS transformation for categorical variables

References
----------
Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for classification trees.
Statistica Sinica, 7, 815-840.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import linalg


def find_qda_split_point(values: np.ndarray, y: np.ndarray) -> float:
    """
    Find the optimal split point using Quadratic Discriminant Analysis.
    
    Algorithm 1 from the QUEST paper. QDA assumes each class follows a normal
    distribution with potentially different variances. The split point is where
    the posterior probabilities are equal.
    
    Mathematical Derivation
    -----------------------
    For binary classification with classes 0 and 1, assume:
    
        Class 0: X ~ N(μ₀, σ₀²)
        Class 1: X ~ N(μ₁, σ₁²)
    
    The QDA discriminant functions are:
    
        δₖ(x) = -½ log|Σₖ| - ½(x - μₖ)ᵀΣₖ⁻¹(x - μₖ) + log πₖ
    
    For the univariate case, setting δ₀(x) = δ₁(x) and solving for x:
    
        (1/σ₁² - 1/σ₀²)x² + 2(μ₀/σ₀² - μ₁/σ₁²)x 
        + (μ₁²/σ₁² - μ₀²/σ₀² + log(σ₁²/σ₀²)) = 0
    
    This is a quadratic equation: Ax² + Bx + C = 0
    
    Special Cases:
    1. Equal variances (σ₀² ≈ σ₁²): Linear discriminant, split at (μ₀ + μ₁)/2
    2. Negative discriminant: No real roots, use midpoint
    3. Multiple roots: Select the one between class means
    
    Parameters
    ----------
    values : np.ndarray of shape (n_samples,)
        Numeric values of the splitting variable (continuous or transformed).
    y : np.ndarray of shape (n_samples,)
        Binary class labels (0 and 1).
    
    Returns
    -------
    float
        The optimal split threshold.
    
    Notes
    -----
    The function is robust to edge cases:
    - Empty classes: returns median of all values
    - Zero variance: adds small epsilon for numerical stability
    - No real roots: falls back to midpoint between means
    
    Examples
    --------
    >>> values = np.array([1.0, 2.0, 3.0, 8.0, 9.0, 10.0])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> threshold = find_qda_split_point(values, y)
    >>> 4.0 < threshold < 7.0  # Should be between class means
    True
    """
    values = np.asarray(values, dtype=np.float64)
    
    # Separate by class
    class_0_mask = (y == 0)
    class_1_mask = (y == 1)
    
    values_0 = values[class_0_mask]
    values_1 = values[class_1_mask]
    
    n_0 = len(values_0)
    n_1 = len(values_1)
    
    # Edge case: empty class
    if n_0 == 0 or n_1 == 0:
        return float(np.median(values))
    
    # Compute class statistics
    mu_0 = np.mean(values_0)
    mu_1 = np.mean(values_1)
    
    # Small epsilon for numerical stability
    epsilon = 1e-10
    
    # Sample variance with Bessel's correction (ddof=1)
    if n_0 > 1:
        var_0 = np.var(values_0, ddof=1)
    else:
        var_0 = epsilon
    
    if n_1 > 1:
        var_1 = np.var(values_1, ddof=1)
    else:
        var_1 = epsilon
    
    # Ensure positive variances
    var_0 = max(var_0, epsilon)
    var_1 = max(var_1, epsilon)
    
    sigma_0 = np.sqrt(var_0)
    sigma_1 = np.sqrt(var_1)
    
    # =========================================================================
    # Special Case: Equal Variances → Linear Discriminant Analysis
    # =========================================================================
    # When variances are approximately equal, the quadratic term vanishes
    # and the optimal split is simply the midpoint between means
    if np.abs(var_0 - var_1) < epsilon * max(var_0, var_1):
        return (mu_0 + mu_1) / 2.0
    
    # =========================================================================
    # General Case: Solve Quadratic Equation
    # =========================================================================
    # Coefficients from QDA decision boundary equation:
    # A*t² + B*t + C = 0
    
    inv_var_0 = 1.0 / var_0
    inv_var_1 = 1.0 / var_1
    
    # Coefficient A = (1/2)(1/σ₀² - 1/σ₁²)
    A = 0.5 * (inv_var_0 - inv_var_1)
    
    # Coefficient B = μ₁/σ₁² - μ₀/σ₀²
    B = mu_1 * inv_var_1 - mu_0 * inv_var_0
    
    # Coefficient C = (1/2)(μ₀²/σ₀² - μ₁²/σ₁²) + log(σ₁/σ₀)
    C = 0.5 * (mu_0**2 * inv_var_0 - mu_1**2 * inv_var_1) + np.log(sigma_1 / sigma_0)
    
    # Discriminant
    discriminant = B**2 - 4*A*C
    
    # No real roots: fall back to midpoint
    if discriminant < 0:
        return (mu_0 + mu_1) / 2.0
    
    # Degenerate linear case (shouldn't happen due to variance check)
    if np.abs(A) < epsilon:
        if np.abs(B) < epsilon:
            return (mu_0 + mu_1) / 2.0
        return -C / B
    
    # Two roots
    sqrt_disc = np.sqrt(discriminant)
    root_1 = (-B + sqrt_disc) / (2 * A)
    root_2 = (-B - sqrt_disc) / (2 * A)
    
    # =========================================================================
    # Select the Physically Meaningful Root
    # =========================================================================
    # Priority:
    # 1. Root between class means (best separation)
    # 2. Root within data range
    # 3. Fallback to midpoint
    
    min_mean = min(mu_0, mu_1)
    max_mean = max(mu_0, mu_1)
    
    data_min = np.min(values)
    data_max = np.max(values)
    
    candidates = []
    
    # Check root_1
    if min_mean <= root_1 <= max_mean:
        candidates.append((0, root_1))  # Priority 0 = between means
    elif data_min <= root_1 <= data_max:
        candidates.append((1, root_1))  # Priority 1 = within data
    
    # Check root_2
    if min_mean <= root_2 <= max_mean:
        candidates.append((0, root_2))
    elif data_min <= root_2 <= data_max:
        candidates.append((1, root_2))
    
    if len(candidates) > 0:
        # Select candidate with best priority (lowest number)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    # Fallback: midpoint between means
    return (mu_0 + mu_1) / 2.0


def compute_crimcoords(
    categorical_column: np.ndarray,
    y: np.ndarray
) -> Dict[Any, float]:
    """
    Compute CRIMCOORDS mapping for a categorical variable.
    
    Algorithm 2 from the QUEST paper. CRIMCOORDS (CRItical Mean COORDinateS)
    transforms categorical values to numeric scores that optimally separate
    the two binary classes.
    
    Mathematical Formulation
    ------------------------
    Let C be the number of categories. The algorithm proceeds as:
    
    1. **One-Hot Encoding**: Create matrix V ∈ ℝⁿˣᶜ
       V[i,j] = 1 if sample i has category j, else 0
    
    2. **Centering**: Ṽ = V - 1ₙv̄ᵀ where v̄ is column means
    
    3. **SVD Decomposition**: Ṽ = UΣWᵀ
       Due to centering, rank(Ṽ) = C-1
    
    4. **Dimension Reduction**: Keep r = C-1 components
       X_reduced = U_r @ Σ_r
    
    5. **LDA in Reduced Space**: 
       - Compute class means: m₀, m₁
       - Within-class scatter: S_W = S₀ + S₁
       - LDA direction: a = S_W⁻¹(m₁ - m₀)
    
    6. **Category Scores**: ξ = W_r @ a
       Each category c gets score ξ[c]
    
    Parameters
    ----------
    categorical_column : np.ndarray of shape (n_samples,)
        Categorical feature values (can be strings, integers, etc.).
    y : np.ndarray of shape (n_samples,)
        Binary class labels (0 and 1).
    
    Returns
    -------
    dict
        Mapping from category value to numeric CRIMCOORDS score.
        Example: {'A': -0.5, 'B': 0.2, 'C': 0.8}
    
    Notes
    -----
    - For binary categories (C=2), simple 0/1 encoding is used.
    - Regularization (epsilon on S_W diagonal) ensures numerical stability.
    - If SVD fails, falls back to ordinal encoding.
    
    The CRIMCOORDS transformation ensures that the split point search
    (via QDA) can use the same machinery for categorical variables
    as for continuous variables.
    
    Examples
    --------
    >>> categories = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> mapping = compute_crimcoords(categories, y)
    >>> # Categories correlated with class 1 get higher scores
    >>> mapping['C'] > mapping['A']
    True
    """
    categories = np.unique(categorical_column)
    n_categories = len(categories)
    n_samples = len(categorical_column)
    
    # Edge case: single category
    if n_categories == 1:
        return {categories[0]: 0.0}
    
    # Edge case: binary category - simple encoding
    if n_categories == 2:
        return {categories[0]: 0.0, categories[1]: 1.0}
    
    # =========================================================================
    # Step 1: One-Hot Encoding → Matrix V
    # =========================================================================
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    V = np.zeros((n_samples, n_categories), dtype=np.float64)
    for i, cat in enumerate(categorical_column):
        V[i, cat_to_idx[cat]] = 1.0
    
    # =========================================================================
    # Step 2: Center V (subtract column means)
    # =========================================================================
    col_means = np.mean(V, axis=0)
    V_centered = V - col_means
    
    # =========================================================================
    # Step 3: SVD Decomposition
    # =========================================================================
    try:
        U, singular_values, Wt = linalg.svd(V_centered, full_matrices=False)
        W = Wt.T  # W is (n_categories × n_components)
    except linalg.LinAlgError:
        # SVD failed; fall back to ordinal encoding
        return {cat: float(i) for i, cat in enumerate(categories)}
    
    # =========================================================================
    # Step 4: Determine effective rank r
    # =========================================================================
    # Due to centering, one singular value should be ~0
    tol = 1e-10 * singular_values[0] if len(singular_values) > 0 else 1e-10
    r = np.sum(singular_values > tol)
    
    if r == 0:
        return {cat: float(i) for i, cat in enumerate(categories)}
    
    # Truncate to rank r
    U_r = U[:, :r]
    Sigma_r = np.diag(singular_values[:r])
    W_r = W[:, :r]
    
    # =========================================================================
    # Step 5: Project to reduced r-dimensional space
    # =========================================================================
    X_reduced = U_r @ Sigma_r
    
    # =========================================================================
    # Step 6: Linear Discriminant Analysis in reduced space
    # =========================================================================
    class_0_mask = (y == 0)
    class_1_mask = (y == 1)
    
    n_0 = np.sum(class_0_mask)
    n_1 = np.sum(class_1_mask)
    
    if n_0 == 0 or n_1 == 0:
        return {cat: float(i) for i, cat in enumerate(categories)}
    
    # Class means in reduced space
    mean_0 = np.mean(X_reduced[class_0_mask], axis=0)
    mean_1 = np.mean(X_reduced[class_1_mask], axis=0)
    
    # Within-class scatter matrix S_W = S₀ + S₁
    X_0_centered = X_reduced[class_0_mask] - mean_0
    X_1_centered = X_reduced[class_1_mask] - mean_1
    
    S_W = X_0_centered.T @ X_0_centered + X_1_centered.T @ X_1_centered
    
    # Regularization for numerical stability
    S_W += np.eye(r) * 1e-6
    
    # LDA direction: a = S_W⁻¹(m₁ - m₀)
    mean_diff = mean_1 - mean_0
    
    try:
        a = linalg.solve(S_W, mean_diff)
    except linalg.LinAlgError:
        # Singular matrix; use pseudo-inverse
        a = linalg.lstsq(S_W, mean_diff)[0]
    
    # =========================================================================
    # Step 7: Compute category scores ξ = W_r @ a
    # =========================================================================
    xi = W_r @ a
    
    # Create mapping dictionary
    crimcoords_map = {cat: float(xi[idx]) for idx, cat in enumerate(categories)}
    
    return crimcoords_map


def get_crimcoords_mapping(
    categorical_column: np.ndarray,
    y: np.ndarray
) -> Dict[Any, float]:
    """
    Alias for compute_crimcoords for backward compatibility.
    
    See compute_crimcoords for full documentation.
    """
    return compute_crimcoords(categorical_column, y)


def transform_categorical_values(
    values: np.ndarray,
    crimcoords_map: Dict[Any, float],
    default_value: float = 0.0
) -> np.ndarray:
    """
    Transform categorical values using a CRIMCOORDS mapping.
    
    Parameters
    ----------
    values : np.ndarray
        Categorical values to transform.
    crimcoords_map : dict
        Mapping from category to numeric score.
    default_value : float
        Value to use for unknown categories.
    
    Returns
    -------
    np.ndarray
        Transformed numeric values.
    
    Examples
    --------
    >>> mapping = {'A': 0.0, 'B': 0.5, 'C': 1.0}
    >>> values = np.array(['A', 'B', 'C', 'B'])
    >>> transform_categorical_values(values, mapping)
    array([0. , 0.5, 1. , 0.5])
    """
    return np.array([crimcoords_map.get(v, default_value) for v in values])


def find_optimal_split(
    feature_values: np.ndarray,
    y: np.ndarray,
    is_categorical: bool = False
) -> Tuple[float, Optional[Dict[Any, float]]]:
    """
    Find the optimal split point for a feature.
    
    This is a convenience function that handles both continuous and
    categorical features, applying the appropriate algorithm.
    
    Parameters
    ----------
    feature_values : np.ndarray
        Values of the feature.
    y : np.ndarray
        Binary class labels.
    is_categorical : bool
        Whether the feature is categorical.
    
    Returns
    -------
    threshold : float
        The optimal split threshold.
    crimcoords_map : dict or None
        CRIMCOORDS mapping if categorical, None otherwise.
    
    Examples
    --------
    >>> # Continuous feature
    >>> values = np.array([1.0, 2.0, 8.0, 9.0])
    >>> y = np.array([0, 0, 1, 1])
    >>> threshold, mapping = find_optimal_split(values, y, is_categorical=False)
    >>> mapping is None
    True
    
    >>> # Categorical feature
    >>> values = np.array(['A', 'A', 'B', 'B'])
    >>> y = np.array([0, 0, 1, 1])
    >>> threshold, mapping = find_optimal_split(values, y, is_categorical=True)
    >>> mapping is not None
    True
    """
    if is_categorical:
        crimcoords_map = compute_crimcoords(feature_values, y)
        transformed_values = transform_categorical_values(feature_values, crimcoords_map)
        threshold = find_qda_split_point(transformed_values, y)
        return threshold, crimcoords_map
    else:
        feature_values = np.asarray(feature_values, dtype=np.float64)
        threshold = find_qda_split_point(feature_values, y)
        return threshold, None
