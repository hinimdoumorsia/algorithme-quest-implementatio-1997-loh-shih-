"""
Automatic Feature Type Detection for QUEST Decision Tree.

This module implements intelligent heuristics to automatically infer
whether features are continuous, categorical, or binary based on:
- NumPy/pandas dtypes
- Cardinality analysis (number of unique values)
- Statistical heuristics

This eliminates the need for manual feature_types specification in most cases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Type aliases
FeatureType = str  # 'continuous', 'categorical', or 'binary'


# =============================================================================
# Configuration Constants
# =============================================================================

# Maximum ratio of unique values to total samples for categorical detection
DEFAULT_MAX_UNIQUE_RATIO = 0.05

# Maximum absolute number of unique values for categorical detection
DEFAULT_MAX_UNIQUE_ABSOLUTE = 20

# Minimum unique values for continuous (below this, consider categorical)
DEFAULT_MIN_UNIQUE_FOR_CONTINUOUS = 10


class FeatureTypeDetector:
    """
    Automatic feature type detector for QUEST decision trees.
    
    This class provides intelligent heuristics to detect whether features
    are continuous, categorical, or binary without requiring manual specification.
    
    The detection algorithm uses multiple signals:
    
    1. **dtype Analysis**: 
       - Object/string dtypes → categorical
       - Boolean dtypes → binary
       - Float dtypes → continuous (unless low cardinality)
       - Integer dtypes → depends on cardinality
    
    2. **Cardinality Analysis**:
       - If unique_values ≤ 2 → binary
       - If unique_values ≤ max_unique_absolute AND 
         unique_ratio ≤ max_unique_ratio → categorical
       - Otherwise → continuous
    
    3. **Value Pattern Analysis**:
       - Integer-like floats (1.0, 2.0) may be categorical
       - Sequential integers (0, 1, 2, ...) are often categorical
    
    Parameters
    ----------
    max_unique_ratio : float
        Maximum ratio of unique values to sample size for categorical.
        Default is 0.05 (5%).
    max_unique_absolute : int
        Maximum number of unique values for categorical detection.
        Default is 20.
    min_unique_for_continuous : int
        Minimum unique values to consider a feature continuous.
        Default is 10.
    
    Attributes
    ----------
    feature_types_ : list of str
        Detected feature types after calling detect().
    feature_info_ : list of dict
        Detailed information about each feature's detection.
    
    Examples
    --------
    >>> detector = FeatureTypeDetector()
    >>> X = np.array([[1.5, 0, 'A'], [2.3, 1, 'B'], [3.1, 0, 'A']], dtype=object)
    >>> types = detector.detect(X)
    >>> types
    ['continuous', 'binary', 'categorical']
    """
    
    def __init__(
        self,
        max_unique_ratio: float = DEFAULT_MAX_UNIQUE_RATIO,
        max_unique_absolute: int = DEFAULT_MAX_UNIQUE_ABSOLUTE,
        min_unique_for_continuous: int = DEFAULT_MIN_UNIQUE_FOR_CONTINUOUS
    ):
        self.max_unique_ratio = max_unique_ratio
        self.max_unique_absolute = max_unique_absolute
        self.min_unique_for_continuous = min_unique_for_continuous
        
        # Fitted attributes
        self.feature_types_: Optional[List[FeatureType]] = None
        self.feature_info_: Optional[List[Dict[str, Any]]] = None
    
    def detect(self, X: np.ndarray) -> List[FeatureType]:
        """
        Detect feature types for all columns in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to analyze.
        
        Returns
        -------
        feature_types : list of str
            List of detected types ('continuous', 'categorical', 'binary')
            for each feature.
        
        Notes
        -----
        After calling this method, detailed detection information is available
        in self.feature_info_.
        """
        n_samples, n_features = X.shape
        
        self.feature_types_ = []
        self.feature_info_ = []
        
        for j in range(n_features):
            feature_col = X[:, j]
            feature_type, info = self._detect_single_feature(
                feature_col, n_samples, j
            )
            self.feature_types_.append(feature_type)
            self.feature_info_.append(info)
        
        return self.feature_types_
    
    def _detect_single_feature(
        self,
        values: np.ndarray,
        n_samples: int,
        feature_index: int
    ) -> Tuple[FeatureType, Dict[str, Any]]:
        """
        Detect the type of a single feature column.
        
        Parameters
        ----------
        values : np.ndarray
            Feature values (1D array).
        n_samples : int
            Total number of samples.
        feature_index : int
            Index of the feature (for logging).
        
        Returns
        -------
        feature_type : str
            Detected type.
        info : dict
            Detailed detection information.
        """
        info: Dict[str, Any] = {
            'feature_index': feature_index,
            'n_samples': n_samples,
            'dtype': str(values.dtype),
            'n_unique': 0,
            'unique_ratio': 0.0,
            'detection_reason': '',
        }
        
        # Handle pandas categorical
        if hasattr(values, 'cat'):
            info['detection_reason'] = 'pandas Categorical dtype'
            info['n_unique'] = len(values.cat.categories)
            return 'categorical', info
        
        # Get unique values
        try:
            unique_values = np.unique(values)
            n_unique = len(unique_values)
        except TypeError:
            # Mixed types that can't be sorted - treat as categorical
            info['detection_reason'] = 'mixed/unsortable types'
            return 'categorical', info
        
        info['n_unique'] = n_unique
        info['unique_ratio'] = n_unique / n_samples if n_samples > 0 else 0.0
        
        # Rule 1: Check dtype first
        dtype = values.dtype
        
        # String/Object dtype → categorical
        if dtype == object or np.issubdtype(dtype, np.str_):
            # Check if all values are numeric strings
            if self._all_numeric_strings(values):
                # Convert and re-analyze
                try:
                    numeric_values = values.astype(float)
                    return self._detect_numeric_feature(
                        numeric_values, n_samples, n_unique, info
                    )
                except (ValueError, TypeError):
                    pass
            
            info['detection_reason'] = 'object/string dtype'
            if n_unique == 2:
                return 'binary', info
            return 'categorical', info
        
        # Boolean dtype → binary
        if dtype == bool or np.issubdtype(dtype, np.bool_):
            info['detection_reason'] = 'boolean dtype'
            return 'binary', info
        
        # Numeric types - analyze cardinality
        return self._detect_numeric_feature(values, n_samples, n_unique, info)
    
    def _detect_numeric_feature(
        self,
        values: np.ndarray,
        n_samples: int,
        n_unique: int,
        info: Dict[str, Any]
    ) -> Tuple[FeatureType, Dict[str, Any]]:
        """
        Detect feature type for numeric features based on cardinality.
        
        Mathematical Justification:
        --------------------------
        For QUEST's statistical tests to work properly:
        - ANOVA F-test requires continuous distributions
        - Chi-square test requires discrete categories
        
        A feature is considered categorical if it has "few" unique values,
        where "few" is defined by both absolute and relative thresholds.
        
        The default threshold of 5% unique values is based on the heuristic
        that categorical variables typically have much lower cardinality than
        continuous variables in practice.
        """
        unique_ratio = n_unique / n_samples if n_samples > 0 else 0.0
        info['unique_ratio'] = unique_ratio
        
        # Rule 2: Binary detection (exactly 2 unique values)
        if n_unique == 2:
            info['detection_reason'] = 'exactly 2 unique values'
            return 'binary', info
        
        # Rule 3: Single value (degenerate case)
        if n_unique == 1:
            info['detection_reason'] = 'single value (degenerate)'
            return 'categorical', info
        
        # Rule 4: Low cardinality → categorical
        # Both conditions must be met to avoid false positives
        is_low_cardinality = (
            n_unique <= self.max_unique_absolute and
            unique_ratio <= self.max_unique_ratio
        )
        
        if is_low_cardinality:
            info['detection_reason'] = (
                f'low cardinality: {n_unique} unique values '
                f'({unique_ratio:.2%} of samples)'
            )
            return 'categorical', info
        
        # Rule 5: Integer dtype with moderate cardinality
        dtype = values.dtype
        if np.issubdtype(dtype, np.integer):
            if n_unique < self.min_unique_for_continuous:
                info['detection_reason'] = (
                    f'integer with {n_unique} unique values '
                    f'(< {self.min_unique_for_continuous} threshold)'
                )
                return 'categorical', info
        
        # Rule 6: Float values that are actually integers
        if np.issubdtype(dtype, np.floating):
            if self._is_integer_valued(values):
                if n_unique < self.min_unique_for_continuous:
                    info['detection_reason'] = (
                        f'float with integer values, {n_unique} unique '
                        f'(< {self.min_unique_for_continuous} threshold)'
                    )
                    return 'categorical', info
        
        # Default: continuous
        info['detection_reason'] = (
            f'high cardinality: {n_unique} unique values '
            f'({unique_ratio:.2%} of samples)'
        )
        return 'continuous', info
    
    def _all_numeric_strings(self, values: np.ndarray) -> bool:
        """Check if all string values are numeric."""
        try:
            for v in values:
                if v is not None and str(v).strip() != '':
                    float(v)
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_integer_valued(self, values: np.ndarray) -> bool:
        """Check if float values are actually integers (e.g., 1.0, 2.0)."""
        # Filter out NaN values
        valid_values = values[~np.isnan(values)] if np.issubdtype(values.dtype, np.floating) else values
        if len(valid_values) == 0:
            return False
        return np.allclose(valid_values, np.round(valid_values))
    
    def get_detection_summary(self) -> str:
        """
        Get a human-readable summary of feature type detection.
        
        Returns
        -------
        summary : str
            Formatted summary of detected feature types.
        """
        if self.feature_types_ is None or self.feature_info_ is None:
            return "No features detected yet. Call detect() first."
        
        lines = ["Feature Type Detection Summary", "=" * 50]
        
        for i, (ftype, info) in enumerate(zip(self.feature_types_, self.feature_info_)):
            lines.append(
                f"Feature {i}: {ftype.upper()}"
                f"\n  - dtype: {info['dtype']}"
                f"\n  - unique values: {info['n_unique']}"
                f"\n  - reason: {info['detection_reason']}"
            )
        
        return "\n".join(lines)


def infer_feature_types(
    X: np.ndarray,
    max_unique_ratio: float = DEFAULT_MAX_UNIQUE_RATIO,
    max_unique_absolute: int = DEFAULT_MAX_UNIQUE_ABSOLUTE
) -> List[FeatureType]:
    """
    Convenience function to infer feature types for a feature matrix.
    
    This is the primary interface for automatic feature type detection.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    max_unique_ratio : float
        Maximum ratio of unique values for categorical detection.
    max_unique_absolute : int
        Maximum absolute unique values for categorical detection.
    
    Returns
    -------
    feature_types : list of str
        List of 'continuous', 'categorical', or 'binary' for each feature.
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> X[:, 2] = np.random.choice([0, 1, 2], size=100)  # Make last col categorical
    >>> types = infer_feature_types(X)
    >>> print(types)
    ['continuous', 'continuous', 'categorical']
    
    Notes
    -----
    Detection Heuristics:
    
    1. **String/Object dtype** → categorical
    2. **Boolean dtype** → binary
    3. **Exactly 2 unique values** → binary
    4. **Low cardinality** (≤ max_unique_absolute AND ≤ max_unique_ratio) → categorical
    5. **Integer with few unique values** → categorical
    6. **Otherwise** → continuous
    
    These heuristics are designed to match common data science conventions
    while providing flexibility through the threshold parameters.
    """
    detector = FeatureTypeDetector(
        max_unique_ratio=max_unique_ratio,
        max_unique_absolute=max_unique_absolute
    )
    return detector.detect(X)


def is_categorical(
    feature_values: np.ndarray,
    max_unique_ratio: float = DEFAULT_MAX_UNIQUE_RATIO,
    max_unique_absolute: int = DEFAULT_MAX_UNIQUE_ABSOLUTE
) -> bool:
    """
    Determine if a single feature should be treated as categorical.
    
    Parameters
    ----------
    feature_values : np.ndarray
        Values of a single feature (1D array).
    max_unique_ratio : float
        Maximum ratio of unique values to sample size.
    max_unique_absolute : int
        Maximum absolute number of unique values.
    
    Returns
    -------
    bool
        True if the feature should be treated as categorical.
    
    Examples
    --------
    >>> continuous_feature = np.random.randn(1000)
    >>> is_categorical(continuous_feature)
    False
    
    >>> categorical_feature = np.random.choice(['A', 'B', 'C'], size=1000)
    >>> is_categorical(categorical_feature)
    True
    """
    feature_values = np.asarray(feature_values)
    
    # Boolean → categorical (binary)
    if feature_values.dtype == bool:
        return True
    
    # String/object dtype - need to check if values are actually numeric
    if feature_values.dtype == object or np.issubdtype(feature_values.dtype, np.str_):
        # Try to convert to float - if it fails, it's categorical
        try:
            numeric_values = np.array([float(v) for v in feature_values])
            # Successfully converted - check cardinality like numeric type
            n_unique = len(np.unique(numeric_values))
            n_total = len(numeric_values)
            if n_unique <= max_unique_absolute or n_unique / n_total <= max_unique_ratio:
                return True
            return False
        except (ValueError, TypeError):
            # Contains non-numeric values - categorical
            return True
    
    # Check cardinality for numeric types
    if np.issubdtype(feature_values.dtype, np.integer):
        n_unique = len(np.unique(feature_values))
        n_total = len(feature_values)
        
        # Low cardinality integers are categorical
        if n_unique <= max_unique_absolute or n_unique / n_total <= max_unique_ratio:
            return True
    
    return False
