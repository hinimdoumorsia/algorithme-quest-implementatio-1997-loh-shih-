"""
QUEST Decision Tree Classifier.

Main QuestTreeClassifier class implementing the QUEST algorithm with
a scikit-learn-compatible API (without sklearn dependency).

References
----------
Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for classification trees.
Statistica Sinica, 7, 815-840.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .node import Node
from .validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_classification_targets,
    check_feature_types,
    ValidationError
)
from .feature_detection import infer_feature_types, FeatureTypeDetector
from .variable_selection import select_best_variable
from .superclass import compute_superclass_labels
from .splits import find_qda_split_point, compute_crimcoords, transform_categorical_values
from .pruning import cost_complexity_prune, compute_subtree_impurity
from .utils import get_max_depth, count_leaves, accuracy_score


# =============================================================================
# Default Hyperparameters (from QUEST paper)
# =============================================================================

DEFAULT_ALPHA = 0.05  # Significance level for hypothesis tests
DEFAULT_MIN_SAMPLES_SPLIT = 10  # Minimum samples to attempt a split
DEFAULT_MIN_SAMPLES_LEAF = 5  # Minimum samples per leaf
DEFAULT_MAX_DEPTH = 20  # Safety limit (pruning is primary regularization)
DEFAULT_CCP_ALPHA = 0.0  # Cost-complexity pruning parameter


class QuestTreeClassifier:
    """
    QUEST Decision Tree Classifier.
    
    Quick, Unbiased, Efficient Statistical Tree - a decision tree algorithm
    that achieves unbiased variable selection through statistical hypothesis
    tests and finds optimal split points using Quadratic Discriminant Analysis.
    
    Key Features
    ------------
    - **Unbiased Variable Selection**: Uses ANOVA/Chi-square tests with
      Bonferroni correction (Algorithm 3)
    - **Multiclass Support**: 2-means clustering on class centroids for
      K > 2 classes (Algorithm 4)
    - **Categorical Variable Handling**: CRIMCOORDS transformation
      (Algorithm 2)
    - **Optimal Split Points**: QDA-based analytical solution (Algorithm 1)
    - **Cost-Complexity Pruning**: CART-style post-pruning
    - **Automatic Feature Type Detection**: No manual specification needed
    
    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for hypothesis tests (ANOVA, Chi-square, Levene).
        Bonferroni correction is applied internally: α* = α/M where M is
        the number of features.
    max_depth : int, default=20
        Maximum depth of the tree. This is a safety limit; cost-complexity
        pruning is the primary regularization method in QUEST.
    min_samples_split : int, default=10
        Minimum number of samples required to attempt a split at a node.
    min_samples_leaf : int, default=5
        Minimum number of samples required in each leaf node.
    ccp_alpha : float, default=0.0
        Cost-complexity pruning parameter (α ≥ 0).
        - α = 0: no pruning (default)
        - larger α: more aggressive pruning
        Optimal value typically found via cross-validation.
    feature_types : list of str, optional
        Manual specification of feature types. Each element should be
        'continuous', 'categorical', or 'binary'. If None (default),
        types are automatically inferred during fit().
    random_state : int, optional
        Random seed for reproducibility. Affects:
        - 2-means clustering in super-class computation
        - Tie-breaking in variable selection
    
    Attributes
    ----------
    root_ : Node
        Root node of the fitted tree.
    n_features_in_ : int
        Number of features seen during fit.
    classes_ : np.ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    n_samples_ : int
        Number of training samples.
    feature_types_ : list of str
        Detected or provided feature types.
    feature_names_in_ : list of str, optional
        Feature names if X was a pandas DataFrame.
    pruning_path_ : list of (float, int)
        Sequence of (alpha, n_leaves) from pruning.
    
    Examples
    --------
    Basic usage with automatic feature detection:
    
    >>> from questtree import QuestTreeClassifier
    >>> import numpy as np
    >>> X = np.random.randn(100, 4)
    >>> y = np.random.choice([0, 1], size=100)
    >>> clf = QuestTreeClassifier()
    >>> clf.fit(X, y)
    QuestTreeClassifier()
    >>> predictions = clf.predict(X)
    >>> accuracy = clf.score(X, y)
    
    With pruning:
    
    >>> clf = QuestTreeClassifier(ccp_alpha=0.01)
    >>> clf.fit(X, y)
    >>> print(f"Leaves: {clf.get_n_leaves()}, Depth: {clf.get_depth()}")
    
    With categorical features:
    
    >>> X = np.column_stack([
    ...     np.random.randn(100, 2),
    ...     np.random.choice(['A', 'B', 'C'], 100)
    ... ])
    >>> clf = QuestTreeClassifier()
    >>> clf.fit(X, y)  # Automatically detects the categorical feature
    
    Notes
    -----
    The QUEST algorithm separates variable selection from split point
    selection to achieve unbiased variable selection. This means that
    variables are selected based on their true predictive power, not
    on the number of possible splits they offer.
    
    See Also
    --------
    questtree.plotting.print_tree : Print tree structure
    questtree.plotting.get_tree_summary : Get tree summary statistics
    """
    
    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        max_depth: int = DEFAULT_MAX_DEPTH,
        min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        ccp_alpha: float = DEFAULT_CCP_ALPHA,
        feature_types: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ) -> None:
        """Initialize QuestTreeClassifier with hyperparameters."""
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.feature_types = feature_types
        self.random_state = random_state
        
        # Fitted attributes (initialized to None)
        self.root_: Optional[Node] = None
        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.n_samples_: Optional[int] = None
        self.feature_types_: Optional[List[str]] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.pruning_path_: Optional[List[Tuple[float, int]]] = None
        self._feature_detector: Optional[FeatureTypeDetector] = None
    
    def fit(
        self,
        X: Any,
        y: Any
    ) -> 'QuestTreeClassifier':
        """
        Build the QUEST decision tree from training data.
        
        The algorithm proceeds in two phases:
        1. **Growing Phase**: Build a large tree with minimal stopping criteria
        2. **Pruning Phase**: Apply cost-complexity pruning to find optimal subtree
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix. Can be:
            - numpy array
            - pandas DataFrame
            - list of lists
        y : array-like of shape (n_samples,)
            Target class labels. Must have at least 2 unique values.
        
        Returns
        -------
        self : QuestTreeClassifier
            Fitted classifier instance.
        
        Raises
        ------
        ValidationError
            If inputs are invalid (wrong shape, missing values, etc.)
        """
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Extract feature names if pandas DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        
        # Validate inputs
        X, y = check_X_y(
            X, y,
            ensure_min_samples=2,
            ensure_min_features=1,
            allow_nan=False
        )
        
        # Store basic info
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        
        # Validate classification targets
        self.classes_, self.n_classes_ = check_classification_targets(y)
        
        # Handle feature types
        if self.feature_types is not None:
            self.feature_types_ = check_feature_types(
                self.feature_types,
                self.n_features_in_
            )
        else:
            # Automatic feature type detection
            self._feature_detector = FeatureTypeDetector()
            self.feature_types_ = self._feature_detector.detect(X)
        
        # =====================================================================
        # Phase 1: Grow Tree
        # =====================================================================
        self.root_ = self._grow_tree(X, y, depth=0)
        
        # =====================================================================
        # Phase 2: Cost-Complexity Pruning
        # =====================================================================
        if self.ccp_alpha > 0:
            self.root_, self.pruning_path_ = cost_complexity_prune(
                self.root_,
                self.ccp_alpha,
                self.n_samples_
            )
        else:
            # Still compute impurities for analysis
            compute_subtree_impurity(self.root_, self.n_samples_)
            self.pruning_path_ = [(0.0, self.root_.n_leaves)]
        
        return self
    
    def _grow_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ) -> Node:
        """
        Recursively grow the decision tree.
        
        This implements the core QUEST algorithm:
        1. Check stopping conditions
        2. Apply Super-Class Clustering if K > 2 (Algorithm 4)
        3. Select best variable (Algorithm 3)
        4. Apply CRIMCOORDS if categorical (Algorithm 2)
        5. Find split point using QDA (Algorithm 1)
        6. Create node and recurse
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for this node.
        y : np.ndarray of shape (n_samples,)
            Original class labels (can have K ≥ 2 unique values).
        depth : int
            Current depth in the tree.
        
        Returns
        -------
        Node
            The constructed tree node.
        """
        n_samples = len(y)
        classes_at_node = np.unique(y)
        n_classes = len(classes_at_node)
        
        # Create node
        node = Node()
        node.depth = depth
        node.n_samples = n_samples
        node.class_counts = {c: int(np.sum(y == c)) for c in classes_at_node}
        
        # =====================================================================
        # Stopping Conditions
        # =====================================================================
        
        # (a) Pure node: all samples belong to one class
        if n_classes == 1:
            node.is_leaf = True
            node.prediction = classes_at_node[0]
            return node
        
        # (b) Maximum depth reached
        if depth >= self.max_depth:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node
        
        # (c) Too few samples to split
        if n_samples < self.min_samples_split:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node
        
        # (d) Cannot create leaves with minimum samples
        if n_samples < 2 * self.min_samples_leaf:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node
        
        # =====================================================================
        # Algorithm 4: Super-Class Clustering
        # Convert K-class to binary for QDA/CRIMCOORDS
        # =====================================================================
        y_binary, group_A, group_B = compute_superclass_labels(
            X, y, self.feature_types_
        )
        node.superclass_groups = (group_A, group_B)
        
        # =====================================================================
        # Algorithm 3: Variable Selection
        # =====================================================================
        best_feature_idx, feature_type = select_best_variable(
            X, y_binary, self.feature_types_, alpha=self.alpha
        )
        
        if best_feature_idx is None:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node
        
        node.feature_index = best_feature_idx
        feature_values = X[:, best_feature_idx]
        
        # =====================================================================
        # Algorithm 2: CRIMCOORDS (if categorical or binary with non-numeric values)
        # =====================================================================
        # Check if we need to use CRIMCOORDS transformation
        # This applies to categorical, binary, or any feature with non-numeric values
        needs_crimcoords = feature_type in ('categorical', 'binary')
        
        # Also check if values can be converted to float (handles object dtype with strings)
        if not needs_crimcoords:
            try:
                # Test if conversion to float is possible
                _ = feature_values.astype(np.float64)
            except (ValueError, TypeError):
                # Values contain non-numeric data, treat as categorical
                needs_crimcoords = True
        
        if needs_crimcoords:
            crimcoords_map = compute_crimcoords(feature_values, y_binary)
            node.crimcoords_map = crimcoords_map
            transformed_values = transform_categorical_values(
                feature_values, crimcoords_map
            )
        else:
            transformed_values = feature_values.astype(np.float64)
            node.crimcoords_map = None
        
        # =====================================================================
        # Algorithm 1: QDA Split Point
        # =====================================================================
        threshold = find_qda_split_point(transformed_values, y_binary)
        node.threshold = threshold
        
        # =====================================================================
        # Split Data
        # =====================================================================
        left_mask = transformed_values <= threshold
        right_mask = ~left_mask
        
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        
        # Check minimum samples per leaf
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node
        
        # =====================================================================
        # Recursive Calls (with ORIGINAL labels)
        # =====================================================================
        node.left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _majority_class(self, y: np.ndarray) -> Any:
        """Return the majority class in y."""
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    def _predict_row(self, row: np.ndarray) -> Any:
        """Predict the class for a single sample."""
        node = self.root_
        
        while not node.is_leaf:
            feature_value = row[node.feature_index]
            
            # Transform if categorical
            if node.crimcoords_map is not None:
                transformed_value = node.crimcoords_map.get(feature_value, 0.0)
            else:
                transformed_value = float(feature_value)
            
            # Traverse
            if transformed_value <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        
        return node.prediction
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        
        Raises
        ------
        ValidationError
            If the classifier is not fitted or X has wrong shape.
        """
        check_is_fitted(self, ['root_', 'n_features_in_'])
        
        X = check_array(X, name="X", ensure_2d=True)
        
        if X.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"X has {X.shape[1]} features, but QuestTreeClassifier "
                f"was fitted with {self.n_features_in_} features."
            )
        
        predictions = [self._predict_row(row) for row in X]
        return np.array(predictions)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        The probability of each class is estimated from the class
        distribution at the leaf node.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities. Order corresponds to self.classes_.
        """
        check_is_fitted(self, ['root_', 'classes_'])
        
        X = check_array(X, name="X", ensure_2d=True)
        
        if X.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"X has {X.shape[1]} features, but QuestTreeClassifier "
                f"was fitted with {self.n_features_in_} features."
            )
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        
        for i, row in enumerate(X):
            node = self.root_
            
            while not node.is_leaf:
                feature_value = row[node.feature_index]
                if node.crimcoords_map is not None:
                    transformed_value = node.crimcoords_map.get(feature_value, 0.0)
                else:
                    transformed_value = float(feature_value)
                
                if transformed_value <= node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            
            # Compute probabilities from class counts
            total = node.n_samples
            for cls, count in node.class_counts.items():
                if cls in class_to_idx:
                    proba[i, class_to_idx[cls]] = count / total
        
        return proba
    
    def score(self, X: Any, y: Any) -> float:
        """
        Return the mean accuracy on the given test data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) compared to y.
        """
        y_pred = self.predict(X)
        y = np.asarray(y)
        return accuracy_score(y, y_pred)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of sub-objects (not applicable here).
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'alpha': self.alpha,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'ccp_alpha': self.ccp_alpha,
            'feature_types': self.feature_types,
            'random_state': self.random_state
        }
    
    def set_params(self, **params: Any) -> 'QuestTreeClassifier':
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : QuestTreeClassifier
            Estimator instance.
        """
        valid_params = self.get_params()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {list(valid_params.keys())}"
                )
            setattr(self, key, value)
        return self
    
    def get_n_leaves(self) -> int:
        """Return the number of leaves in the tree."""
        if self.root_ is None:
            return 0
        return self.root_.n_leaves
    
    def get_depth(self) -> int:
        """Return the maximum depth of the tree."""
        if self.root_ is None:
            return 0
        return get_max_depth(self.root_)
    
    def get_feature_importances(self) -> Dict[int, float]:
        """
        Compute feature importances based on split frequency.
        
        Returns
        -------
        importances : dict
            Mapping from feature index to importance (0-1 scale).
        """
        from .utils import compute_feature_importances
        
        check_is_fitted(self, ['root_', 'n_features_in_'])
        return compute_feature_importances(self.root_, self.n_features_in_)
    
    def __repr__(self) -> str:
        """Return string representation of the classifier."""
        params = []
        if self.alpha != DEFAULT_ALPHA:
            params.append(f"alpha={self.alpha}")
        if self.max_depth != DEFAULT_MAX_DEPTH:
            params.append(f"max_depth={self.max_depth}")
        if self.min_samples_split != DEFAULT_MIN_SAMPLES_SPLIT:
            params.append(f"min_samples_split={self.min_samples_split}")
        if self.min_samples_leaf != DEFAULT_MIN_SAMPLES_LEAF:
            params.append(f"min_samples_leaf={self.min_samples_leaf}")
        if self.ccp_alpha != DEFAULT_CCP_ALPHA:
            params.append(f"ccp_alpha={self.ccp_alpha}")
        if self.random_state is not None:
            params.append(f"random_state={self.random_state}")
        
        return f"QuestTreeClassifier({', '.join(params)})"
    
    def __str__(self) -> str:
        """Return human-readable string representation."""
        return self.__repr__()
