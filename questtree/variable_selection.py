"""
Variable Selection (Algorithm 3) for QUEST Decision Tree.

Implements unbiased variable selection using statistical hypothesis tests:
- ANOVA F-test for continuous variables
- Chi-square test for categorical variables
- Levene's test as a fallback for variance differences

References
----------
Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for classification trees.
Statistica Sinica, 7, 815-840.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from .feature_detection import is_categorical


def calc_anova_p_value(feature_values: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the p-value from a one-way ANOVA F-test.
    
    Tests whether the means of a continuous variable differ significantly
    between classes. This is the primary test for continuous features in QUEST.
    
    Mathematical Formulation
    ------------------------
    Hypothesis test:
        H₀: μ₁ = μ₂ = ... = μₖ (class means are equal)
        H₁: At least one mean differs
    
    F-statistic:
                     Σₖ nₖ(x̄ₖ - x̄)² / (K-1)     SS_between / df_between
        F = ─────────────────────────────── = ─────────────────────────
            Σₖ Σᵢ∈Cₖ (xᵢ - x̄ₖ)² / (n-K)       SS_within / df_within
    
    where:
        - x̄ₖ = mean of variable in class k
        - x̄ = grand mean
        - nₖ = samples in class k
        - K = number of classes
        - n = total samples
    
    Under H₀: F ~ F(K-1, n-K)
    
    Parameters
    ----------
    feature_values : np.ndarray of shape (n_samples,)
        Values of the continuous feature.
    y : np.ndarray of shape (n_samples,)
        Class labels.
    
    Returns
    -------
    float
        P-value from the F-test. Smaller values indicate stronger
        evidence against H₀ (i.e., the feature is predictive).
    
    Notes
    -----
    This is equivalent to scipy.stats.f_oneway but implemented from
    scratch to avoid dependencies and provide transparency.
    
    Examples
    --------
    >>> # Strong separation
    >>> x = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(5, 1, 50)])
    >>> y = np.array([0]*50 + [1]*50)
    >>> p = calc_anova_p_value(x, y)
    >>> p < 0.01  # Should be highly significant
    True
    """
    classes = np.unique(y)
    k = len(classes)  # Number of groups
    n = len(y)        # Total samples
    
    # Grand mean
    grand_mean = np.mean(feature_values)
    
    # Compute group statistics
    group_means = []
    group_sizes = []
    
    for c in classes:
        mask = (y == c)
        group_values = feature_values[mask]
        group_means.append(np.mean(group_values))
        group_sizes.append(len(group_values))
    
    # Between-Group Sum of Squares (SS_B)
    # SS_B = Σₖ nₖ(x̄ₖ - x̄)²
    ss_between = 0.0
    for j in range(k):
        ss_between += group_sizes[j] * (group_means[j] - grand_mean) ** 2
    
    # Within-Group Sum of Squares (SS_W)
    # SS_W = Σₖ Σᵢ∈Cₖ (xᵢ - x̄ₖ)²
    ss_within = 0.0
    for j, c in enumerate(classes):
        mask = (y == c)
        group_values = feature_values[mask]
        ss_within += np.sum((group_values - group_means[j]) ** 2)
    
    # Degrees of freedom
    df_between = k - 1
    df_within = n - k
    
    # Edge cases
    if df_within <= 0 or ss_within == 0:
        return 1.0  # Cannot reject H₀
    
    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # F-statistic
    f_statistic = ms_between / ms_within
    
    # P-value from F-distribution
    p_value = 1.0 - stats.f.cdf(f_statistic, df_between, df_within)
    
    return float(p_value)


def calc_chisquare_p_value(feature_values: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the p-value from a Chi-square test of independence.
    
    Tests whether a categorical variable is independent of the class label.
    This is the primary test for categorical features in QUEST.
    
    Mathematical Formulation
    ------------------------
    Hypothesis test:
        H₀: Variable and class are independent
        H₁: Variable and class are dependent
    
    Chi-square statistic:
                    (Oᵢⱼ - Eᵢⱼ)²
        χ² = Σᵢ Σⱼ ─────────────
                       Eᵢⱼ
    
    where:
        - Oᵢⱼ = observed frequency of (category i, class j)
        - Eᵢⱼ = expected frequency = (row_i_total × col_j_total) / n
    
    Under H₀: χ² ~ χ²((C-1)(K-1))
    where C = number of categories, K = number of classes
    
    Parameters
    ----------
    feature_values : np.ndarray of shape (n_samples,)
        Values of the categorical feature.
    y : np.ndarray of shape (n_samples,)
        Class labels.
    
    Returns
    -------
    float
        P-value from the Chi-square test.
    
    Examples
    --------
    >>> # Strong association
    >>> x = np.array(['A']*30 + ['B']*30 + ['A']*10 + ['B']*30)
    >>> y = np.array([0]*60 + [1]*40)
    >>> p = calc_chisquare_p_value(x, y)
    >>> p < 0.05  # Should show association
    True
    """
    categories = np.unique(feature_values)
    classes = np.unique(y)
    
    n_categories = len(categories)
    n_classes = len(classes)
    n_total = len(y)
    
    # Build contingency table
    # Rows = categories, Columns = classes
    observed = np.zeros((n_categories, n_classes))
    
    for i, cat in enumerate(categories):
        for j, cls in enumerate(classes):
            observed[i, j] = np.sum((feature_values == cat) & (y == cls))
    
    # Row and column totals
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    
    # Expected frequencies under independence
    # E_ij = (row_i × col_j) / n
    expected = np.outer(row_totals, col_totals) / n_total
    
    # Avoid division by zero
    epsilon = 1e-10
    expected = np.maximum(expected, epsilon)
    
    # Chi-square statistic
    chi_sq = np.sum((observed - expected) ** 2 / expected)
    
    # Degrees of freedom
    df = (n_categories - 1) * (n_classes - 1)
    
    if df <= 0:
        return 1.0
    
    # P-value from Chi-square distribution
    p_value = 1.0 - stats.chi2.cdf(chi_sq, df)
    
    return float(p_value)


def calc_levene_p_value(feature_values: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the p-value from Levene's test for equality of variances.
    
    Used as a fallback test when ANOVA/Chi-square tests are not significant.
    Levene's test detects differences in variance (spread) rather than
    mean (location).
    
    Mathematical Formulation (Brown-Forsythe variant)
    -------------------------------------------------
    1. Transform data: zᵢⱼ = |xᵢⱼ - median(xⱼ)|
       where xⱼ is the j-th group
    
    2. Apply ANOVA to z values:
                 Σⱼ nⱼ(z̄ⱼ - z̄)² / (K-1)
        W = ─────────────────────────────
            Σⱼ Σᵢ (zᵢⱼ - z̄ⱼ)² / (n-K)
    
    Using medians (Brown-Forsythe) is more robust to non-normality
    than using means (original Levene).
    
    Parameters
    ----------
    feature_values : np.ndarray of shape (n_samples,)
        Values of the continuous feature.
    y : np.ndarray of shape (n_samples,)
        Class labels.
    
    Returns
    -------
    float
        P-value from Levene's test.
    
    Notes
    -----
    Levene's test is useful when class distributions have similar means
    but different variances. In QUEST, this serves as a secondary test
    when ANOVA doesn't find significance.
    
    Examples
    --------
    >>> # Different variances, same mean
    >>> x = np.concatenate([np.random.normal(0, 0.5, 50), np.random.normal(0, 3, 50)])
    >>> y = np.array([0]*50 + [1]*50)
    >>> p = calc_levene_p_value(x, y)
    >>> p < 0.05  # Should detect variance difference
    True
    """
    classes = np.unique(y)
    k = len(classes)
    n = len(y)
    
    # Transform: z_ij = |x_ij - median(group_j)|
    z_values = np.zeros_like(feature_values, dtype=np.float64)
    
    for c in classes:
        mask = (y == c)
        group_values = feature_values[mask]
        group_median = np.median(group_values)
        z_values[mask] = np.abs(group_values - group_median)
    
    # Apply ANOVA to z-values
    grand_mean_z = np.mean(z_values)
    
    group_means_z = []
    group_sizes = []
    
    for c in classes:
        mask = (y == c)
        group_z = z_values[mask]
        group_means_z.append(np.mean(group_z))
        group_sizes.append(len(group_z))
    
    # Between-group SS
    ss_between = 0.0
    for j in range(k):
        ss_between += group_sizes[j] * (group_means_z[j] - grand_mean_z) ** 2
    
    # Within-group SS
    ss_within = 0.0
    for j, c in enumerate(classes):
        mask = (y == c)
        group_z = z_values[mask]
        ss_within += np.sum((group_z - group_means_z[j]) ** 2)
    
    # Degrees of freedom
    df_between = k - 1
    df_within = n - k
    
    if df_within <= 0 or ss_within == 0:
        return 1.0
    
    # F-statistic (Levene's W)
    f_statistic = (ss_between / df_between) / (ss_within / df_within)
    
    # P-value
    p_value = 1.0 - stats.f.cdf(f_statistic, df_between, df_within)
    
    return float(p_value)


def select_best_variable(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Tuple[Optional[int], Optional[str]]:
    """
    Select the best splitting variable using Algorithm 3 from QUEST.
    
    This implements the unbiased variable selection procedure:
    
    Phase 1 (Primary Tests):
        1. For continuous variables: ANOVA F-test
        2. For categorical variables: Chi-square test
        3. Apply Bonferroni correction: α* = α/M
        4. If any p-value < α*, select variable with smallest p-value
    
    Phase 2 (Fallback - Levene's Test):
        5. If no variable passes Phase 1, apply Levene's test to
           continuous variables (tests variance differences)
        6. Apply Bonferroni correction to Levene p-values
        7. Select from significant Levene tests
    
    Phase 3 (Last Resort):
        8. If still nothing significant, select variable with
           smallest p-value from Phase 1
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Binary class labels (0 and 1).
    feature_types : list of str, optional
        List of 'continuous' or 'categorical' for each feature.
        If None, types are inferred automatically.
    alpha : float
        Significance level for hypothesis tests (before Bonferroni).
    
    Returns
    -------
    best_feature_idx : int or None
        Index of the selected feature, or None if no feature found.
    feature_type : str or None
        Type of the selected feature ('continuous' or 'categorical').
    
    Notes
    -----
    The use of statistical tests with Bonferroni correction ensures
    unbiased selection - variables with more possible splits don't
    have an unfair advantage.
    
    Examples
    --------
    >>> X = np.column_stack([
    ...     np.random.normal(0, 1, 100),  # noise
    ...     np.concatenate([np.random.normal(0, 1, 50), np.random.normal(3, 1, 50)])  # signal
    ... ])
    >>> y = np.array([0]*50 + [1]*50)
    >>> idx, ftype = select_best_variable(X, y)
    >>> idx == 1  # Should select the informative feature
    True
    """
    n_samples, n_features = X.shape
    
    # Infer feature types if not provided
    if feature_types is None:
        feature_types = []
        for j in range(n_features):
            if is_categorical(X[:, j]):
                feature_types.append('categorical')
            else:
                feature_types.append('continuous')
    
    # Bonferroni correction
    alpha_corrected = alpha / n_features
    
    # =========================================================================
    # Phase 1: ANOVA / Chi-square tests
    # =========================================================================
    p_values = []
    
    for j in range(n_features):
        feature_values = X[:, j]
        
        if feature_types[j] == 'continuous':
            feature_values = feature_values.astype(np.float64)
            p_val = calc_anova_p_value(feature_values, y)
        else:
            p_val = calc_chisquare_p_value(feature_values, y)
        
        p_values.append(p_val)
    
    p_values = np.array(p_values)
    
    # Check for significant variables
    significant_mask = p_values < alpha_corrected
    
    if np.any(significant_mask):
        best_idx = int(np.argmin(p_values))
        return best_idx, feature_types[best_idx]
    
    # =========================================================================
    # Phase 2: Levene's test for continuous variables
    # =========================================================================
    continuous_indices = [j for j in range(n_features) 
                         if feature_types[j] == 'continuous']
    
    if len(continuous_indices) == 0:
        # No continuous variables; use smallest chi-square p-value
        best_idx = int(np.argmin(p_values))
        return best_idx, feature_types[best_idx]
    
    levene_p_values = np.ones(n_features)
    
    for j in continuous_indices:
        feature_values = X[:, j].astype(np.float64)
        levene_p_values[j] = calc_levene_p_value(feature_values, y)
    
    # Bonferroni correction for Levene tests
    n_continuous = len(continuous_indices)
    alpha_levene = alpha / n_continuous if n_continuous > 0 else alpha
    
    levene_significant = levene_p_values < alpha_levene
    
    if np.any(levene_significant):
        best_idx = int(np.argmin(levene_p_values))
        return best_idx, feature_types[best_idx]
    
    # =========================================================================
    # Phase 3: No significant tests - use smallest p-value
    # =========================================================================
    best_idx = int(np.argmin(p_values))
    return best_idx, feature_types[best_idx]


def compute_all_test_statistics(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: Optional[List[str]] = None
) -> dict:
    """
    Compute all test statistics for variable selection diagnostics.
    
    This function is useful for understanding why a particular variable
    was selected and for debugging.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Class labels.
    feature_types : list of str, optional
        Feature types.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'anova_p_values': list of ANOVA p-values for continuous features
        - 'chisq_p_values': list of Chi-square p-values for categorical features
        - 'levene_p_values': list of Levene p-values for continuous features
        - 'feature_types': inferred or provided feature types
        - 'selected_feature': index of selected feature
    """
    n_samples, n_features = X.shape
    
    if feature_types is None:
        feature_types = []
        for j in range(n_features):
            if is_categorical(X[:, j]):
                feature_types.append('categorical')
            else:
                feature_types.append('continuous')
    
    anova_p_values = []
    chisq_p_values = []
    levene_p_values = []
    
    for j in range(n_features):
        feature_values = X[:, j]
        
        if feature_types[j] == 'continuous':
            feature_values = feature_values.astype(np.float64)
            anova_p_values.append(calc_anova_p_value(feature_values, y))
            chisq_p_values.append(np.nan)
            levene_p_values.append(calc_levene_p_value(feature_values, y))
        else:
            anova_p_values.append(np.nan)
            chisq_p_values.append(calc_chisquare_p_value(feature_values, y))
            levene_p_values.append(np.nan)
    
    selected_idx, _ = select_best_variable(X, y, feature_types)
    
    return {
        'anova_p_values': anova_p_values,
        'chisq_p_values': chisq_p_values,
        'levene_p_values': levene_p_values,
        'feature_types': feature_types,
        'selected_feature': selected_idx
    }
