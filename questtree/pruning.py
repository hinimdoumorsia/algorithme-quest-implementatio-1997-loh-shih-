"""
Cost-Complexity Pruning for QUEST Decision Tree.

Implements CART-style minimal cost-complexity pruning as described in:
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).
  Classification and Regression Trees. Wadsworth.

References
----------
Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for classification trees.
Statistica Sinica, 7, 815-840.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .node import Node


def compute_node_impurity(node: 'Node', n_total: int) -> float:
    """
    Compute the misclassification rate (impurity) for a node.
    
    For cost-complexity pruning, the impurity of node t is:
    
        R(t) = (nₜ / n) × (1 - max_k(pₖ))
    
    where:
        - nₜ = number of samples at node t
        - n = total samples in tree
        - pₖ = proportion of class k at node t
    
    This is the weighted misclassification rate, representing the
    contribution of this node to the overall tree error if it were a leaf.
    
    Parameters
    ----------
    node : Node
        The tree node.
    n_total : int
        Total number of samples in the training set.
    
    Returns
    -------
    float
        Node impurity (weighted misclassification rate).
    
    Examples
    --------
    >>> from questtree.node import Node
    >>> node = Node()
    >>> node.n_samples = 100
    >>> node.class_counts = {0: 80, 1: 20}
    >>> impurity = compute_node_impurity(node, n_total=1000)
    >>> # R(t) = (100/1000) * (1 - 0.8) = 0.02
    >>> abs(impurity - 0.02) < 1e-10
    True
    """
    if node.n_samples == 0:
        return 0.0
    
    # Proportion of samples at this node
    node_weight = node.n_samples / n_total
    
    # Majority class proportion
    if len(node.class_counts) == 0:
        max_proportion = 1.0
    else:
        max_count = max(node.class_counts.values())
        max_proportion = max_count / node.n_samples
    
    # Misclassification rate: 1 - majority proportion
    misclass_rate = 1.0 - max_proportion
    
    # Weighted impurity
    return node_weight * misclass_rate


def compute_subtree_impurity(node: 'Node', n_total: int) -> Tuple[float, int]:
    """
    Recursively compute the total impurity of the subtree rooted at node.
    
    For a leaf: R(Tₜ) = R(t)
    For an internal node: R(Tₜ) = R(T_left) + R(T_right)
    
    Also computes |Tₜ| (number of leaves in subtree).
    
    This function updates the node's impurity attributes in-place:
        - node.impurity: R(t) - impurity if this were a leaf
        - node.subtree_impurity: R(Tₜ) - total impurity of subtree
        - node.n_leaves: |Tₜ| - number of leaves
    
    Parameters
    ----------
    node : Node
        The tree node.
    n_total : int
        Total number of samples in training set.
    
    Returns
    -------
    subtree_impurity : float
        Total impurity of the subtree.
    n_leaves : int
        Number of leaves in the subtree.
    
    Notes
    -----
    This must be called before pruning to initialize all impurity values.
    """
    if node.is_leaf:
        impurity = compute_node_impurity(node, n_total)
        node.impurity = impurity
        node.subtree_impurity = impurity
        node.n_leaves = 1
        return impurity, 1
    
    # Recurse on children
    left_impurity, left_leaves = compute_subtree_impurity(node.left_child, n_total)
    right_impurity, right_leaves = compute_subtree_impurity(node.right_child, n_total)
    
    # This node's impurity if it were a leaf
    node.impurity = compute_node_impurity(node, n_total)
    
    # Subtree impurity (sum of children)
    node.subtree_impurity = left_impurity + right_impurity
    node.n_leaves = left_leaves + right_leaves
    
    return node.subtree_impurity, node.n_leaves


def compute_effective_alpha(node: 'Node') -> float:
    """
    Compute the effective alpha for a node.
    
    The effective alpha (or critical value) represents the complexity
    parameter at which pruning this node's subtree becomes beneficial.
    
    Mathematical Formula
    --------------------
    
        α_eff(t) = (R(t) - R(Tₜ)) / (|Tₜ| - 1)
    
    where:
        - R(t) = impurity if this node were a leaf
        - R(Tₜ) = total impurity of subtree rooted at t
        - |Tₜ| = number of leaves in subtree
    
    Interpretation:
        - Numerator: increase in error from pruning
        - Denominator: decrease in complexity from pruning
        - Ratio: error-per-leaf-removed
    
    Parameters
    ----------
    node : Node
        The tree node.
    
    Returns
    -------
    float
        Effective alpha value. Returns infinity for leaf nodes.
    
    Notes
    -----
    In weakest-link pruning, we prune the node with smallest α_eff first.
    This gives a sequence of nested subtrees optimal for each complexity level.
    """
    if node.is_leaf or node.n_leaves <= 1:
        return np.inf
    
    numerator = node.impurity - node.subtree_impurity
    denominator = node.n_leaves - 1
    
    if denominator <= 0:
        return np.inf
    
    return numerator / denominator


def find_min_alpha_node(
    node: 'Node',
    min_alpha: float = np.inf,
    min_node: Optional['Node'] = None
) -> Tuple[float, Optional['Node']]:
    """
    Find the internal node with the smallest effective alpha.
    
    Traverses the tree to find the "weakest link" - the node that
    should be pruned first in the cost-complexity pruning sequence.
    
    Parameters
    ----------
    node : Node
        Current node in traversal.
    min_alpha : float
        Current minimum alpha found.
    min_node : Node or None
        Node with current minimum alpha.
    
    Returns
    -------
    min_alpha : float
        Smallest effective alpha in the subtree.
    min_node : Node or None
        Node with smallest effective alpha.
    """
    if node.is_leaf:
        return min_alpha, min_node
    
    # Compute effective alpha for this node
    alpha = compute_effective_alpha(node)
    
    if alpha < min_alpha:
        min_alpha = alpha
        min_node = node
    
    # Recurse on children
    min_alpha, min_node = find_min_alpha_node(node.left_child, min_alpha, min_node)
    min_alpha, min_node = find_min_alpha_node(node.right_child, min_alpha, min_node)
    
    return min_alpha, min_node


def prune_node(node: 'Node') -> None:
    """
    Convert an internal node to a leaf (prune its subtree).
    
    This operation:
    1. Sets is_leaf = True
    2. Removes references to children
    3. Updates n_leaves to 1
    4. Sets prediction to majority class
    
    Parameters
    ----------
    node : Node
        The node to prune.
    
    Notes
    -----
    This modifies the node in-place. The subtree rooted at this node
    is effectively deleted.
    """
    node.is_leaf = True
    node.left_child = None
    node.right_child = None
    node.n_leaves = 1
    node.subtree_impurity = node.impurity
    
    # Set prediction to majority class
    if len(node.class_counts) > 0:
        node.prediction = max(node.class_counts.keys(), 
                             key=lambda k: node.class_counts[k])


def cost_complexity_prune(
    root: 'Node',
    ccp_alpha: float,
    n_total: int
) -> Tuple['Node', List[Tuple[float, int]]]:
    """
    Apply cost-complexity pruning to the tree.
    
    This implements the minimal cost-complexity pruning algorithm:
    
    Algorithm
    ---------
    1. Compute impurities for all nodes
    2. Find node t with minimum α_eff(t)
    3. If α_eff(t) ≤ ccp_alpha:
       - Prune node t (convert to leaf)
       - Repeat from step 1
    4. Stop when α_eff > ccp_alpha for all nodes
    
    The sequence of pruned trees is optimal in the sense that for
    any α, the tree minimizing R_α(T) = R(T) + α|T| is in this sequence.
    
    Parameters
    ----------
    root : Node
        Root of the tree to prune.
    ccp_alpha : float
        Complexity parameter (α ≥ 0).
        - α = 0: no pruning
        - larger α: more aggressive pruning
    n_total : int
        Total number of training samples.
    
    Returns
    -------
    root : Node
        Root of the pruned tree (same object, modified in-place).
    pruning_path : list of (float, int)
        Sequence of (alpha, n_leaves) showing the pruning progression.
    
    Raises
    ------
    ValueError
        If ccp_alpha is negative.
    
    Examples
    --------
    >>> # Prune with alpha = 0.01
    >>> pruned_root, path = cost_complexity_prune(root, ccp_alpha=0.01, n_total=1000)
    >>> # Path shows how tree was pruned
    >>> for alpha, n_leaves in path:
    ...     print(f"α={alpha:.4f}: {n_leaves} leaves")
    
    Notes
    -----
    The pruning is done in-place on the tree structure. If you need to
    preserve the original tree, make a deep copy before calling this function.
    """
    if ccp_alpha < 0:
        raise ValueError("ccp_alpha must be non-negative")
    
    pruning_path: List[Tuple[float, int]] = []
    
    # Initial computation of all impurities
    compute_subtree_impurity(root, n_total)
    pruning_path.append((0.0, root.n_leaves))
    
    if ccp_alpha == 0:
        return root, pruning_path
    
    # Iteratively prune until no beneficial pruning remains
    while True:
        # Recompute impurities after structural changes
        compute_subtree_impurity(root, n_total)
        
        # Find node with minimum effective alpha
        min_alpha, min_node = find_min_alpha_node(root)
        
        if min_node is None or min_alpha > ccp_alpha:
            # No more nodes to prune at this alpha level
            break
        
        # Prune this node
        prune_node(min_node)
        
        # Recompute and record
        compute_subtree_impurity(root, n_total)
        pruning_path.append((min_alpha, root.n_leaves))
        
        # Check if tree is now just the root
        if root.is_leaf:
            break
    
    return root, pruning_path


def generate_pruning_path(
    root: 'Node',
    n_total: int
) -> List[Tuple[float, int, 'Node']]:
    """
    Generate the complete pruning path (sequence of subtrees).
    
    This creates a sequence of (alpha, n_leaves, tree) tuples where
    increasing alpha produces smaller trees. Useful for cross-validation
    to find optimal alpha.
    
    Parameters
    ----------
    root : Node
        Root of the fully grown tree.
    n_total : int
        Total number of training samples.
    
    Returns
    -------
    path : list of (float, int, Node)
        Sequence of (alpha, n_leaves, tree_copy) sorted by increasing alpha.
    
    Notes
    -----
    This function creates deep copies of the tree at each step to preserve
    the entire sequence. This can be memory-intensive for large trees.
    
    For just selecting an alpha, use cost_complexity_prune with cross-validation
    on the training data instead.
    """
    path: List[Tuple[float, int, 'Node']] = []
    
    # Work on a deep copy to preserve original
    current_root = root.deep_copy()
    
    # Initial state
    compute_subtree_impurity(current_root, n_total)
    path.append((0.0, current_root.n_leaves, current_root.deep_copy()))
    
    # Keep pruning until only root remains
    while not current_root.is_leaf:
        compute_subtree_impurity(current_root, n_total)
        min_alpha, min_node = find_min_alpha_node(current_root)
        
        if min_node is None:
            break
        
        prune_node(min_node)
        compute_subtree_impurity(current_root, n_total)
        
        path.append((min_alpha, current_root.n_leaves, current_root.deep_copy()))
    
    return path


def get_optimal_alpha_cv(
    X: np.ndarray,
    y: np.ndarray,
    tree_builder_func,
    n_folds: int = 5,
    alphas: Optional[List[float]] = None
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    Find optimal ccp_alpha using cross-validation.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Class labels.
    tree_builder_func : callable
        Function that takes (X, y, ccp_alpha) and returns a fitted tree.
    n_folds : int
        Number of cross-validation folds.
    alphas : list of float, optional
        Alpha values to try. If None, generates from initial tree's pruning path.
    
    Returns
    -------
    best_alpha : float
        Optimal alpha value (one-SE rule).
    cv_results : list of (alpha, mean_score, std_score)
        Cross-validation results for each alpha.
    
    Notes
    -----
    Uses the "one standard error rule": selects the simplest model
    (largest alpha) whose score is within one standard error of the best.
    """
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    fold_size = n_samples // n_folds
    
    if alphas is None:
        # Get alphas from initial tree's pruning path
        initial_tree = tree_builder_func(X, y, 0.0)
        _, path = cost_complexity_prune(initial_tree.root_, 0.0, n_samples)
        alphas = [0.0] + [a for a, _ in path[1:]]
        alphas = sorted(set(alphas))
    
    cv_results: List[Tuple[float, float, float]] = []
    
    for alpha in alphas:
        fold_scores = []
        
        for fold in range(n_folds):
            # Split data
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
            
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and evaluate
            tree = tree_builder_func(X_train, y_train, alpha)
            predictions = tree.predict(X_test)
            accuracy = np.mean(predictions == y_test)
            fold_scores.append(accuracy)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        cv_results.append((alpha, mean_score, std_score))
    
    # Find best alpha using one-SE rule
    best_idx = np.argmax([s for _, s, _ in cv_results])
    best_score = cv_results[best_idx][1]
    best_std = cv_results[best_idx][2]
    threshold = best_score - best_std
    
    # Find largest alpha within one SE of best
    for alpha, score, _ in reversed(cv_results):
        if score >= threshold:
            return alpha, cv_results
    
    return cv_results[best_idx][0], cv_results
