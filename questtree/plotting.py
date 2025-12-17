"""
Visualization Utilities for QUEST Decision Tree.

Provides functions for printing, displaying, and visualizing decision trees.
All visualization code is isolated in this module to keep the core algorithm clean.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from .classifier import QuestTreeClassifier
    from .node import Node


def print_tree(
    clf: 'QuestTreeClassifier',
    node: Optional['Node'] = None,
    indent: str = "",
    file: TextIO = sys.stdout,
    feature_names: Optional[List[str]] = None
) -> None:
    """
    Print a text representation of the QUEST decision tree.
    
    Displays the tree structure with:
    - Feature used at each split
    - Threshold values
    - Class distributions
    - Super-class groupings (for multiclass)
    
    Parameters
    ----------
    clf : QuestTreeClassifier
        Fitted classifier.
    node : Node, optional
        Starting node (default: root).
    indent : str
        Indentation string for formatting.
    file : TextIO
        Output file (default: stdout).
    feature_names : list of str, optional
        Names for features. If None, uses indices or clf.feature_names_in_.
    
    Examples
    --------
    >>> clf = QuestTreeClassifier()
    >>> clf.fit(X, y)
    >>> print_tree(clf)
    QUEST Decision Tree Structure:
      Classes: [0, 1]
      Leaves: 5
      Depth: 3
    ============================================================
    [Node] Feature 0 (continuous)
           Threshold: 2.5000
           n_samples: 100
      |-- Left (<=):
      |   [LEAF] Predict: 0 | n=45 | dist=(0:40, 1:5)
      |-- Right (>):
          [Node] Feature 2 (categorical)
          ...
    """
    # Get feature names
    if feature_names is None:
        if hasattr(clf, 'feature_names_in_') and clf.feature_names_in_ is not None:
            feature_names = clf.feature_names_in_
    
    if node is None:
        node = clf.root_
        print("QUEST Decision Tree Structure:", file=file)
        print(f"  Classes: {list(clf.classes_)}", file=file)
        print(f"  Leaves: {clf.get_n_leaves()}", file=file)
        print(f"  Depth: {clf.get_depth()}", file=file)
        print("=" * 60, file=file)
    
    if node.is_leaf:
        class_dist = ", ".join([f"{k}:{v}" for k, v in sorted(node.class_counts.items())])
        print(
            f"{indent}[LEAF] Predict: {node.prediction} | "
            f"n={node.n_samples} | dist=({class_dist})",
            file=file
        )
    else:
        feature_idx = node.feature_index
        feature_type = clf.feature_types_[feature_idx] if clf.feature_types_ else "unknown"
        
        # Get feature name
        if feature_names and feature_idx < len(feature_names):
            feature_str = f"{feature_names[feature_idx]} (feature {feature_idx})"
        else:
            feature_str = f"Feature {feature_idx}"
        
        # Show super-class grouping for multiclass
        if clf.n_classes_ > 2 and node.superclass_groups is not None:
            group_A, group_B = node.superclass_groups
            print(f"{indent}[Node] {feature_str} ({feature_type})", file=file)
            print(f"{indent}       Super-classes: {sorted(group_A)} vs {sorted(group_B)}", file=file)
        else:
            print(f"{indent}[Node] {feature_str} ({feature_type})", file=file)
        
        # Show threshold
        if node.crimcoords_map is not None:
            print(f"{indent}       CRIMCOORDS threshold: {node.threshold:.4f}", file=file)
            # Show category mapping
            sorted_cats = sorted(node.crimcoords_map.items(), key=lambda x: x[1])
            left_cats = [str(c) for c, v in sorted_cats if v <= node.threshold]
            right_cats = [str(c) for c, v in sorted_cats if v > node.threshold]
            if left_cats:
                print(f"{indent}       Left categories: {left_cats}", file=file)
            if right_cats:
                print(f"{indent}       Right categories: {right_cats}", file=file)
        else:
            print(f"{indent}       Threshold: {node.threshold:.4f}", file=file)
        
        print(f"{indent}       n_samples: {node.n_samples}", file=file)
        
        # Print children
        print(f"{indent}  |-- Left (<=):", file=file)
        print_tree(clf, node.left_child, indent + "  |   ", file, feature_names)
        
        print(f"{indent}  |-- Right (>):", file=file)
        print_tree(clf, node.right_child, indent + "      ", file, feature_names)


def get_tree_summary(clf: 'QuestTreeClassifier') -> Dict[str, Any]:
    """
    Get a summary dictionary of the tree structure.
    
    Parameters
    ----------
    clf : QuestTreeClassifier
        Fitted classifier.
    
    Returns
    -------
    summary : dict
        Dictionary containing:
        - n_leaves: number of leaf nodes
        - max_depth: maximum tree depth
        - n_classes: number of classes
        - classes: list of class labels
        - n_features: number of features
        - n_samples_trained: training samples
        - ccp_alpha: pruning parameter used
        - feature_types: detected feature types
        - pruning_path: sequence of (alpha, n_leaves)
    
    Examples
    --------
    >>> summary = get_tree_summary(clf)
    >>> print(f"Tree has {summary['n_leaves']} leaves and depth {summary['max_depth']}")
    """
    return {
        'n_leaves': clf.get_n_leaves(),
        'max_depth': clf.get_depth(),
        'n_classes': clf.n_classes_,
        'classes': list(clf.classes_) if clf.classes_ is not None else [],
        'n_features': clf.n_features_in_,
        'n_samples_trained': clf.n_samples_,
        'ccp_alpha': clf.ccp_alpha,
        'feature_types': clf.feature_types_,
        'pruning_path': clf.pruning_path_,
    }


def get_feature_importances(
    clf: 'QuestTreeClassifier',
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute feature importances based on split frequency.
    
    Parameters
    ----------
    clf : QuestTreeClassifier
        Fitted classifier.
    feature_names : list of str, optional
        Names for features. If None, uses indices.
    
    Returns
    -------
    importances : dict
        Mapping from feature name/index to importance (0-1 scale).
    
    Examples
    --------
    >>> importances = get_feature_importances(clf)
    >>> for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
    ...     print(f"{name}: {imp:.4f}")
    """
    raw_importances = clf.get_feature_importances()
    
    # Get feature names
    if feature_names is None:
        if hasattr(clf, 'feature_names_in_') and clf.feature_names_in_ is not None:
            feature_names = clf.feature_names_in_
    
    if feature_names:
        return {feature_names[i]: v for i, v in raw_importances.items()}
    else:
        return {f"feature_{i}": v for i, v in raw_importances.items()}


def tree_to_rules(
    clf: 'QuestTreeClassifier',
    feature_names: Optional[List[str]] = None
) -> List[str]:
    """
    Extract decision rules from the tree.
    
    Returns a list of if-then rules, one per leaf node.
    
    Parameters
    ----------
    clf : QuestTreeClassifier
        Fitted classifier.
    feature_names : list of str, optional
        Names for features.
    
    Returns
    -------
    rules : list of str
        List of decision rules in readable format.
    
    Examples
    --------
    >>> rules = tree_to_rules(clf)
    >>> for rule in rules:
    ...     print(rule)
    IF feature_0 <= 2.5 AND feature_1 > 1.0 THEN class = 1
    """
    # Get feature names
    if feature_names is None:
        if hasattr(clf, 'feature_names_in_') and clf.feature_names_in_ is not None:
            feature_names = clf.feature_names_in_
    
    rules = []
    
    def _extract_rules(node: 'Node', conditions: List[str]) -> None:
        if node.is_leaf:
            if conditions:
                rule = "IF " + " AND ".join(conditions) + f" THEN class = {node.prediction}"
            else:
                rule = f"THEN class = {node.prediction}"
            rules.append(rule)
            return
        
        # Get feature name
        feat_idx = node.feature_index
        if feature_names and feat_idx < len(feature_names):
            feat_name = feature_names[feat_idx]
        else:
            feat_name = f"feature_{feat_idx}"
        
        # Left branch condition
        if node.crimcoords_map is not None:
            # Categorical: show which categories go left
            left_cats = [str(c) for c, v in node.crimcoords_map.items() 
                        if v <= node.threshold]
            left_cond = f"{feat_name} in {left_cats}"
            right_cond = f"{feat_name} not in {left_cats}"
        else:
            left_cond = f"{feat_name} <= {node.threshold:.4f}"
            right_cond = f"{feat_name} > {node.threshold:.4f}"
        
        _extract_rules(node.left_child, conditions + [left_cond])
        _extract_rules(node.right_child, conditions + [right_cond])
    
    _extract_rules(clf.root_, [])
    return rules


def export_text(
    clf: 'QuestTreeClassifier',
    feature_names: Optional[List[str]] = None,
    spacing: int = 3
) -> str:
    """
    Build a text report showing the rules of a decision tree.
    
    Similar to sklearn's export_text for compatibility.
    
    Parameters
    ----------
    clf : QuestTreeClassifier
        Fitted classifier.
    feature_names : list of str, optional
        Names for features.
    spacing : int
        Number of spaces for each level of indentation.
    
    Returns
    -------
    report : str
        Text representation of the tree.
    """
    import io
    output = io.StringIO()
    
    # Get feature names
    if feature_names is None:
        if hasattr(clf, 'feature_names_in_') and clf.feature_names_in_ is not None:
            feature_names = clf.feature_names_in_
    
    def _build_text(node: 'Node', depth: int) -> None:
        indent = " " * (depth * spacing)
        
        if node.is_leaf:
            class_dist = ", ".join([f"{k}: {v}" for k, v in sorted(node.class_counts.items())])
            output.write(f"{indent}class: {node.prediction} [{class_dist}]\n")
            return
        
        feat_idx = node.feature_index
        if feature_names and feat_idx < len(feature_names):
            feat_name = feature_names[feat_idx]
        else:
            feat_name = f"feature_{feat_idx}"
        
        if node.crimcoords_map is not None:
            # Categorical split
            left_cats = [str(c) for c, v in node.crimcoords_map.items() 
                        if v <= node.threshold]
            output.write(f"{indent}|--- {feat_name} in {left_cats}\n")
            _build_text(node.left_child, depth + 1)
            output.write(f"{indent}|--- {feat_name} not in {left_cats}\n")
            _build_text(node.right_child, depth + 1)
        else:
            # Continuous split
            output.write(f"{indent}|--- {feat_name} <= {node.threshold:.4f}\n")
            _build_text(node.left_child, depth + 1)
            output.write(f"{indent}|--- {feat_name} > {node.threshold:.4f}\n")
            _build_text(node.right_child, depth + 1)
    
    _build_text(clf.root_, 0)
    return output.getvalue()


def plot_pruning_path(
    clf: 'QuestTreeClassifier',
    ax: Optional[Any] = None
) -> Any:
    """
    Plot the cost-complexity pruning path.
    
    Shows how the number of leaves decreases as alpha increases.
    
    Parameters
    ----------
    clf : QuestTreeClassifier
        Fitted classifier with pruning_path_.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    
    Notes
    -----
    Requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_pruning_path")
    
    if clf.pruning_path_ is None:
        raise ValueError("Classifier does not have pruning_path_. "
                        "Make sure ccp_alpha > 0 or generate pruning path manually.")
    
    alphas = [a for a, _ in clf.pruning_path_]
    n_leaves = [l for _, l in clf.pruning_path_]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(alphas, n_leaves, marker='o', linestyle='-', markersize=6)
    ax.set_xlabel('Alpha (complexity parameter)')
    ax.set_ylabel('Number of leaves')
    ax.set_title('QUEST Tree Cost-Complexity Pruning Path')
    ax.grid(True, alpha=0.3)
    
    # Mark the selected alpha
    ax.axvline(x=clf.ccp_alpha, color='r', linestyle='--', 
               label=f'Selected α = {clf.ccp_alpha}')
    ax.legend()
    
    return ax
