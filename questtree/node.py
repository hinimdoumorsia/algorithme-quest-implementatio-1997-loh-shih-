"""
Node Data Structure for QUEST Decision Tree.

Defines the Node class representing individual nodes in the decision tree,
supporting both internal (splitting) nodes and leaf nodes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple


class Node:
    """
    A node in the QUEST decision tree.
    
    This class represents both internal (splitting) nodes and leaf (terminal) nodes.
    It stores all information necessary for tree construction, prediction,
    and cost-complexity pruning.
    
    For internal nodes:
    - Stores the splitting feature index and threshold
    - For categorical features, stores the CRIMCOORDS mapping
    - Stores references to left and right child nodes
    - Stores super-class grouping for multiclass problems
    
    For leaf nodes:
    - Stores the class prediction
    - Is marked with is_leaf=True
    
    Attributes
    ----------
    feature_index : int or None
        Index of the splitting feature. None for leaf nodes.
    threshold : float or None
        Split threshold value. For continuous features, this is the raw threshold.
        For categorical features (with CRIMCOORDS), this is the threshold in the
        transformed space.
    crimcoords_map : dict or None
        Mapping from category values to CRIMCOORDS scores for categorical splits.
        None for continuous features or leaf nodes.
    left_child : Node or None
        Left child node (samples with feature value <= threshold).
    right_child : Node or None
        Right child node (samples with feature value > threshold).
    is_leaf : bool
        True if this is a leaf (terminal) node.
    prediction : any
        Class prediction for leaf nodes. The predicted class label.
    depth : int
        Depth of this node in the tree (root has depth 0).
    n_samples : int
        Number of training samples that reached this node.
    class_counts : dict
        Dictionary mapping class labels to sample counts at this node.
        Example: {0: 45, 1: 55} for binary classification.
    superclass_groups : tuple of (set, set) or None
        For multiclass problems (K > 2), stores the super-class partition used
        at this node. Format: (group_A_classes, group_B_classes).
        This is used by Algorithm 4 (Super-Class Clustering).
    impurity : float
        Misclassification rate if this node were a leaf.
        R(t) = (n_t / n) * (1 - max_k(p_k)) where p_k is the proportion of class k.
    subtree_impurity : float
        Total weighted impurity of the subtree rooted at this node.
        R(T_t) = sum of impurities of all leaves in subtree.
    n_leaves : int
        Number of leaf nodes in the subtree rooted at this node.
        |T_t| in the cost-complexity formula.
    
    Mathematical Notes
    ------------------
    Cost-Complexity Measure:
        For pruning, we use the cost-complexity criterion:
        
        R_α(T) = R(T) + α|T|
        
        where:
        - R(T) is the total misclassification cost (sum of leaf impurities)
        - |T| is the number of leaves
        - α is the complexity parameter
        
        The effective alpha for a node t is:
        
        α_eff(t) = (R(t) - R(T_t)) / (|T_t| - 1)
        
        This measures the "cost per leaf removed" if we prune the subtree.
    
    Examples
    --------
    Creating a leaf node:
    
    >>> leaf = Node()
    >>> leaf.is_leaf = True
    >>> leaf.prediction = 1
    >>> leaf.n_samples = 50
    >>> leaf.class_counts = {0: 10, 1: 40}
    
    Creating an internal node:
    
    >>> node = Node()
    >>> node.feature_index = 2
    >>> node.threshold = 0.5
    >>> node.left_child = leaf
    >>> node.right_child = another_leaf
    """
    
    __slots__ = [
        'feature_index', 'threshold', 'crimcoords_map',
        'left_child', 'right_child', 'is_leaf', 'prediction',
        'depth', 'n_samples', 'class_counts', 'superclass_groups',
        'impurity', 'subtree_impurity', 'n_leaves'
    ]
    
    def __init__(self) -> None:
        """Initialize a new Node with default values."""
        # Splitting information
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.crimcoords_map: Optional[Dict[Any, float]] = None
        
        # Tree structure
        self.left_child: Optional[Node] = None
        self.right_child: Optional[Node] = None
        self.is_leaf: bool = False
        
        # Prediction and metadata
        self.prediction: Optional[Any] = None
        self.depth: int = 0
        self.n_samples: int = 0
        self.class_counts: Dict[Any, int] = {}
        
        # Multiclass support (Algorithm 4: Super-Class Clustering)
        self.superclass_groups: Optional[Tuple[Set[Any], Set[Any]]] = None
        
        # Cost-complexity pruning attributes
        self.impurity: float = 0.0
        self.subtree_impurity: float = 0.0
        self.n_leaves: int = 1
    
    def __repr__(self) -> str:
        """Return a string representation of the node."""
        if self.is_leaf:
            return f"Leaf(prediction={self.prediction}, n_samples={self.n_samples})"
        else:
            threshold_str = f"{self.threshold:.4f}" if self.threshold is not None else "None"
            is_categorical = self.crimcoords_map is not None
            type_str = "categorical" if is_categorical else "continuous"
            return (
                f"Node(feature={self.feature_index}, "
                f"threshold={threshold_str}, "
                f"type={type_str}, "
                f"n_samples={self.n_samples})"
            )
    
    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return self.__repr__()
    
    @property
    def is_internal(self) -> bool:
        """Check if this is an internal (non-leaf) node."""
        return not self.is_leaf
    
    @property
    def is_categorical_split(self) -> bool:
        """Check if this node uses a categorical split (CRIMCOORDS)."""
        return self.crimcoords_map is not None
    
    @property
    def majority_class(self) -> Optional[Any]:
        """
        Get the majority class at this node.
        
        Returns
        -------
        class_label : any or None
            The class with the most samples at this node, or None if empty.
        """
        if not self.class_counts:
            return None
        return max(self.class_counts.keys(), key=lambda k: self.class_counts[k])
    
    @property
    def class_distribution(self) -> Dict[Any, float]:
        """
        Get the class probability distribution at this node.
        
        Returns
        -------
        distribution : dict
            Dictionary mapping class labels to probabilities.
        """
        if self.n_samples == 0:
            return {}
        return {k: v / self.n_samples for k, v in self.class_counts.items()}
    
    def get_effective_alpha(self) -> float:
        """
        Compute the effective alpha for cost-complexity pruning.
        
        The effective alpha is the complexity parameter at which pruning
        this node's subtree becomes beneficial.
        
        Mathematical Formula (from CART):
        
        α_eff(t) = (R(t) - R(T_t)) / (|T_t| - 1)
        
        where:
        - R(t) = impurity if this node were a leaf
        - R(T_t) = total impurity of subtree
        - |T_t| = number of leaves in subtree
        
        Returns
        -------
        float
            Effective alpha value. Returns infinity for leaf nodes.
        
        Notes
        -----
        This is used in the weakest-link pruning algorithm. Nodes with
        smaller effective alpha are pruned first.
        """
        import numpy as np
        
        if self.is_leaf or self.n_leaves <= 1:
            return np.inf
        
        numerator = self.impurity - self.subtree_impurity
        denominator = self.n_leaves - 1
        
        if denominator <= 0:
            return np.inf
        
        return numerator / denominator
    
    def copy(self) -> 'Node':
        """
        Create a shallow copy of this node.
        
        Returns
        -------
        Node
            A new Node with the same attributes (but sharing child references).
        """
        new_node = Node()
        new_node.feature_index = self.feature_index
        new_node.threshold = self.threshold
        new_node.crimcoords_map = self.crimcoords_map
        new_node.left_child = self.left_child
        new_node.right_child = self.right_child
        new_node.is_leaf = self.is_leaf
        new_node.prediction = self.prediction
        new_node.depth = self.depth
        new_node.n_samples = self.n_samples
        new_node.class_counts = self.class_counts.copy()
        new_node.superclass_groups = self.superclass_groups
        new_node.impurity = self.impurity
        new_node.subtree_impurity = self.subtree_impurity
        new_node.n_leaves = self.n_leaves
        return new_node
    
    def deep_copy(self) -> 'Node':
        """
        Create a deep copy of this node and its entire subtree.
        
        Returns
        -------
        Node
            A new Node with recursively copied children.
        """
        new_node = self.copy()
        
        if self.left_child is not None:
            new_node.left_child = self.left_child.deep_copy()
        if self.right_child is not None:
            new_node.right_child = self.right_child.deep_copy()
        
        # Deep copy mutable attributes
        if self.crimcoords_map is not None:
            new_node.crimcoords_map = self.crimcoords_map.copy()
        if self.superclass_groups is not None:
            new_node.superclass_groups = (
                self.superclass_groups[0].copy(),
                self.superclass_groups[1].copy()
            )
        
        return new_node
