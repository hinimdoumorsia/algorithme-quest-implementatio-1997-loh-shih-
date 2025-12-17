"""
Input Validation for QUEST Decision Tree.

Provides validation utilities for checking inputs without sklearn dependencies.

This module implements research-grade input validation with clear error messages
for debugging and ensuring data quality before tree construction.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np


class ValidationError(ValueError):
    """Custom exception for validation errors with detailed messages."""
    pass


def check_array(
    array: Any,
    name: str = "array",
    dtype: Optional[Union[type, str]] = None,
    ensure_2d: bool = False,
    allow_nd: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 0,
    allow_nan: bool = False,
    copy: bool = False
) -> np.ndarray:
    """
    Validate and convert input array to numpy ndarray.
    
    This function performs comprehensive validation on input arrays,
    ensuring they meet the requirements for QUEST tree training and prediction.
    
    Parameters
    ----------
    array : array-like
        Input array to validate. Can be list, tuple, numpy array, or pandas DataFrame.
    name : str
        Name of the array for error messages.
    dtype : type or str, optional
        Desired data type. If None, dtype is inferred.
    ensure_2d : bool
        If True, ensure the array is 2-dimensional.
    allow_nd : bool
        If True, allow arrays with more than 2 dimensions.
    ensure_min_samples : int
        Minimum number of samples (rows) required.
    ensure_min_features : int
        Minimum number of features (columns) required. Only checked if ensure_2d=True.
    allow_nan : bool
        If True, allow NaN values in the array.
    copy : bool
        If True, return a copy of the array.
    
    Returns
    -------
    np.ndarray
        Validated numpy array.
    
    Raises
    ------
    ValidationError
        If validation fails with a descriptive error message.
    
    Examples
    --------
    >>> X = [[1, 2], [3, 4]]
    >>> X_validated = check_array(X, name="X", ensure_2d=True)
    >>> X_validated.shape
    (2, 2)
    """
    # Handle None input
    if array is None:
        raise ValidationError(f"{name} cannot be None.")
    
    # Try to convert to numpy array
    try:
        # Handle pandas DataFrame/Series
        if hasattr(array, 'values'):
            array = array.values
        
        result = np.asarray(array)
        
        if copy:
            result = result.copy()
            
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"Unable to convert {name} to numpy array. "
            f"Ensure it is a valid array-like object. Error: {e}"
        )
    
    # Check for empty array
    if result.size == 0:
        raise ValidationError(f"{name} cannot be empty.")
    
    # Check dimensions
    if ensure_2d:
        if result.ndim == 1:
            # Reshape 1D array to column vector
            result = result.reshape(-1, 1)
        elif result.ndim != 2:
            raise ValidationError(
                f"{name} must be 2-dimensional. Got {result.ndim} dimensions."
            )
    
    if not allow_nd and result.ndim > 2:
        raise ValidationError(
            f"{name} must be at most 2-dimensional. Got {result.ndim} dimensions."
        )
    
    # Check minimum samples
    n_samples = result.shape[0]
    if n_samples < ensure_min_samples:
        raise ValidationError(
            f"{name} requires at least {ensure_min_samples} samples. "
            f"Got {n_samples} samples."
        )
    
    # Check minimum features (only for 2D arrays)
    if ensure_2d and ensure_min_features > 0:
        n_features = result.shape[1]
        if n_features < ensure_min_features:
            raise ValidationError(
                f"{name} requires at least {ensure_min_features} features. "
                f"Got {n_features} features."
            )
    
    # Check for NaN values
    if not allow_nan:
        # Check only for numeric types
        if np.issubdtype(result.dtype, np.floating):
            if np.any(np.isnan(result)):
                nan_count = np.sum(np.isnan(result))
                raise ValidationError(
                    f"{name} contains {nan_count} NaN value(s). "
                    f"QUEST does not support missing values. "
                    f"Please impute or remove missing data before fitting."
                )
    
    # Convert dtype if specified
    if dtype is not None:
        try:
            result = result.astype(dtype)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Unable to convert {name} to dtype {dtype}. Error: {e}"
            )
    
    return result


def check_X_y(
    X: Any,
    y: Any,
    allow_multioutput: bool = False,
    ensure_min_samples: int = 1,
    ensure_min_features: int = 1,
    allow_nan: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate feature matrix X and target array y for consistency.
    
    This function ensures X and y have compatible shapes and valid values
    for supervised learning with the QUEST algorithm.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    allow_multioutput : bool
        If True, allow y to be 2-dimensional (multi-output).
    ensure_min_samples : int
        Minimum number of samples required.
    ensure_min_features : int
        Minimum number of features required.
    allow_nan : bool
        If True, allow NaN values in X.
    
    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Validated feature matrix.
    y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Validated target array.
    
    Raises
    ------
    ValidationError
        If validation fails.
    
    Examples
    --------
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> y = [0, 1, 0]
    >>> X_val, y_val = check_X_y(X, y)
    >>> X_val.shape, y_val.shape
    ((3, 2), (3,))
    """
    # Validate X
    X = check_array(
        X,
        name="X",
        ensure_2d=True,
        ensure_min_samples=ensure_min_samples,
        ensure_min_features=ensure_min_features,
        allow_nan=allow_nan
    )
    
    # Validate y
    y = check_array(y, name="y", ensure_min_samples=ensure_min_samples)
    
    # Ensure y is 1D unless multi-output is allowed
    if not allow_multioutput:
        if y.ndim > 1:
            if y.shape[1] == 1:
                y = y.ravel()
            else:
                raise ValidationError(
                    f"y should be 1-dimensional. Got shape {y.shape}. "
                    f"For multi-output classification, set allow_multioutput=True."
                )
    
    # Check that X and y have the same number of samples
    n_samples_X = X.shape[0]
    n_samples_y = y.shape[0]
    
    if n_samples_X != n_samples_y:
        raise ValidationError(
            f"X and y have inconsistent number of samples. "
            f"X has {n_samples_X} samples, y has {n_samples_y} samples."
        )
    
    return X, y


def check_is_fitted(estimator: Any, attributes: List[str]) -> None:
    """
    Check if an estimator is fitted by verifying required attributes exist.
    
    Parameters
    ----------
    estimator : object
        The estimator instance to check.
    attributes : list of str
        List of attribute names that should exist if the estimator is fitted.
    
    Raises
    ------
    ValidationError
        If the estimator is not fitted (missing required attributes).
    
    Examples
    --------
    >>> class MyEstimator:
    ...     def fit(self, X, y):
    ...         self.root_ = "fitted"
    ...         return self
    >>> est = MyEstimator()
    >>> check_is_fitted(est, ['root_'])  # Raises error
    >>> est.fit([[1]], [0])
    >>> check_is_fitted(est, ['root_'])  # No error
    """
    missing = []
    for attr in attributes:
        if not hasattr(estimator, attr) or getattr(estimator, attr) is None:
            missing.append(attr)
    
    if missing:
        estimator_name = type(estimator).__name__
        raise ValidationError(
            f"This {estimator_name} instance is not fitted yet. "
            f"Call 'fit' with appropriate arguments before using this estimator. "
            f"Missing attributes: {missing}"
        )


def check_classification_targets(y: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Validate that y contains valid classification targets.
    
    Parameters
    ----------
    y : np.ndarray
        Target array to validate.
    
    Returns
    -------
    classes : np.ndarray
        Unique class labels sorted.
    n_classes : int
        Number of unique classes.
    
    Raises
    ------
    ValidationError
        If y contains invalid classification targets.
    
    Notes
    -----
    QUEST requires at least 2 classes for classification.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    
    if n_classes < 2:
        raise ValidationError(
            f"Classification requires at least 2 distinct classes. "
            f"Found {n_classes} class(es): {list(classes)}. "
            f"Check that your target variable contains multiple classes."
        )
    
    # Check for reasonable number of samples per class
    min_samples_per_class = np.min([np.sum(y == c) for c in classes])
    if min_samples_per_class < 1:
        raise ValidationError(
            f"At least one class has no samples. "
            f"Ensure all classes have at least one sample."
        )
    
    return classes, n_classes


def check_feature_types(
    feature_types: Optional[List[str]],
    n_features: int
) -> Optional[List[str]]:
    """
    Validate user-provided feature types.
    
    Parameters
    ----------
    feature_types : list of str or None
        List of feature types ('continuous', 'categorical', 'binary').
        If None, feature types will be auto-detected.
    n_features : int
        Expected number of features.
    
    Returns
    -------
    feature_types : list of str or None
        Validated feature types or None.
    
    Raises
    ------
    ValidationError
        If feature_types is invalid.
    """
    if feature_types is None:
        return None
    
    # Check length matches
    if len(feature_types) != n_features:
        raise ValidationError(
            f"feature_types length ({len(feature_types)}) must match "
            f"number of features ({n_features})."
        )
    
    # Check valid type values
    valid_types = {'continuous', 'categorical', 'binary'}
    for i, ft in enumerate(feature_types):
        if ft not in valid_types:
            raise ValidationError(
                f"Invalid feature type '{ft}' at index {i}. "
                f"Valid types are: {valid_types}"
            )
    
    return list(feature_types)


def check_sample_weight(
    sample_weight: Optional[Any],
    n_samples: int
) -> Optional[np.ndarray]:
    """
    Validate sample weights.
    
    Parameters
    ----------
    sample_weight : array-like or None
        Sample weights. If None, uniform weights are assumed.
    n_samples : int
        Expected number of samples.
    
    Returns
    -------
    sample_weight : np.ndarray or None
        Validated sample weights.
    
    Raises
    ------
    ValidationError
        If sample_weight is invalid.
    
    Notes
    -----
    Currently, QUEST does not implement weighted samples, but this validation
    is provided for future compatibility.
    """
    if sample_weight is None:
        return None
    
    sample_weight = check_array(
        sample_weight,
        name="sample_weight",
        ensure_min_samples=n_samples
    )
    
    if sample_weight.ndim != 1:
        raise ValidationError(
            f"sample_weight must be 1-dimensional. Got {sample_weight.ndim} dimensions."
        )
    
    if len(sample_weight) != n_samples:
        raise ValidationError(
            f"sample_weight length ({len(sample_weight)}) must match "
            f"number of samples ({n_samples})."
        )
    
    if np.any(sample_weight < 0):
        raise ValidationError("sample_weight cannot contain negative values.")
    
    return sample_weight.astype(np.float64)
