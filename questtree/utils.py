"""
Utility Functions for QUEST Decision Tree.

General utility functions used across the module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .node import Node


def count_leaves(node: 'Node') -> int:
    """
    Count the number of leaf nodes in a tree.
    
    Parameters
    ----------
    node : Node
        Root node of the tree or subtree.
    
    Returns
    -------
    int
        Number of leaf nodes.
    """
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return count_leaves(node.left_child) + count_leaves(node.right_child)


def get_max_depth(node: 'Node') -> int:
    """
    Get the maximum depth of a tree.
    
    Parameters
    ----------
    node : Node
        Root node of the tree.
    
    Returns
    -------
    int
        Maximum depth (root has depth 0).
    """
    if node is None:
        return -1
    if node.is_leaf:
        return node.depth
    return max(get_max_depth(node.left_child), get_max_depth(node.right_child))


def count_nodes(node: 'Node') -> int:
    """
    Count total number of nodes (internal + leaf) in a tree.
    
    Parameters
    ----------
    node : Node
        Root node.
    
    Returns
    -------
    int
        Total node count.
    """
    if node is None:
        return 0
    return 1 + count_nodes(node.left_child) + count_nodes(node.right_child)


def collect_leaf_predictions(node: 'Node') -> List[Any]:
    """
    Collect all predictions from leaf nodes.
    
    Parameters
    ----------
    node : Node
        Root node.
    
    Returns
    -------
    list
        List of predictions from all leaves.
    """
    if node is None:
        return []
    if node.is_leaf:
        return [node.prediction]
    
    left_preds = collect_leaf_predictions(node.left_child)
    right_preds = collect_leaf_predictions(node.right_child)
    return left_preds + right_preds


def collect_split_features(node: 'Node') -> List[int]:
    """
    Collect all feature indices used in splits.
    
    Parameters
    ----------
    node : Node
        Root node.
    
    Returns
    -------
    list of int
        List of feature indices (may contain duplicates).
    """
    if node is None or node.is_leaf:
        return []
    
    features = [node.feature_index]
    features.extend(collect_split_features(node.left_child))
    features.extend(collect_split_features(node.right_child))
    return features


def compute_feature_importances(
    node: 'Node',
    n_features: int
) -> Dict[int, float]:
    """
    Compute feature importances based on split frequency.
    
    Importance is measured as the proportion of splits using each feature.
    
    Parameters
    ----------
    node : Node
        Root node.
    n_features : int
        Total number of features.
    
    Returns
    -------
    dict
        Mapping from feature index to importance (0-1 scale).
    """
    split_features = collect_split_features(node)
    
    if len(split_features) == 0:
        return {i: 0.0 for i in range(n_features)}
    
    # Count occurrences
    counts: Dict[int, int] = {}
    for f in split_features:
        counts[f] = counts.get(f, 0) + 1
    
    total = sum(counts.values())
    
    # Normalize to proportions
    importances = {i: 0.0 for i in range(n_features)}
    for f, count in counts.items():
        importances[f] = count / total
    
    return importances


def tree_to_dict(node: 'Node', feature_names: Optional[List[str]] = None) -> Dict:
    """
    Convert a tree to a dictionary representation.
    
    Useful for serialization and inspection.
    
    Parameters
    ----------
    node : Node
        Root node.
    feature_names : list of str, optional
        Names for features.
    
    Returns
    -------
    dict
        Nested dictionary representing tree structure.
    """
    if node is None:
        return {}
    
    result = {
        'is_leaf': node.is_leaf,
        'n_samples': node.n_samples,
        'depth': node.depth,
        'class_counts': dict(node.class_counts),
    }
    
    if node.is_leaf:
        result['prediction'] = node.prediction
    else:
        feature_idx = node.feature_index
        if feature_names and feature_idx is not None:
            result['feature_name'] = feature_names[feature_idx]
        result['feature_index'] = feature_idx
        result['threshold'] = node.threshold
        result['is_categorical_split'] = node.crimcoords_map is not None
        
        if node.superclass_groups:
            result['superclass_groups'] = [
                list(node.superclass_groups[0]),
                list(node.superclass_groups[1])
            ]
        
        result['left'] = tree_to_dict(node.left_child, feature_names)
        result['right'] = tree_to_dict(node.right_child, feature_names)
    
    return result


def apply_to_nodes(node: 'Node', func, **kwargs) -> List:
    """
    Apply a function to all nodes in the tree.
    
    Parameters
    ----------
    node : Node
        Root node.
    func : callable
        Function taking (node, **kwargs) and returning a value.
    **kwargs
        Additional arguments passed to func.
    
    Returns
    -------
    list
        Results from applying func to each node.
    """
    if node is None:
        return []
    
    results = [func(node, **kwargs)]
    results.extend(apply_to_nodes(node.left_child, func, **kwargs))
    results.extend(apply_to_nodes(node.right_child, func, **kwargs))
    return results


def majority_class(y: np.ndarray) -> Any:
    """
    Find the majority class in an array of labels.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels.
    
    Returns
    -------
    any
        The most frequent class label.
    """
    unique, counts = np.unique(y, return_counts=True)
    return unique[np.argmax(counts)]


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    
    Returns
    -------
    float
        Accuracy score (0-1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, List]:
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    
    Returns
    -------
    matrix : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix where entry (i, j) is count of samples
        with true label i and predicted label j.
    classes : list
        Ordered list of class labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = sorted(set(y_true) | set(y_pred))
    n_classes = len(classes)
    
    label_to_idx = {c: i for i, c in enumerate(classes)}
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        matrix[label_to_idx[true], label_to_idx[pred]] += 1
    
    return matrix, classes


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate a text classification report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    
    Returns
    -------
    str
        Formatted classification report with precision, recall, F1.
    """
    matrix, classes = confusion_matrix(y_true, y_pred)
    
    lines = ["Classification Report", "=" * 60]
    lines.append(f"{'Class':>15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    lines.append("-" * 60)
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for i, cls in enumerate(classes):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        support = matrix[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        
        lines.append(f"{str(cls):>15} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f} {support:>10}")
    
    lines.append("-" * 60)
    
    # Macro average
    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    lines.append(f"{'macro avg':>15} {macro_p:>12.4f} {macro_r:>12.4f} {macro_f1:>12.4f} {sum(supports):>10}")
    
    # Weighted average
    total = sum(supports)
    weighted_p = sum(p * s for p, s in zip(precisions, supports)) / total if total > 0 else 0
    weighted_r = sum(r * s for r, s in zip(recalls, supports)) / total if total > 0 else 0
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total if total > 0 else 0
    lines.append(f"{'weighted avg':>15} {weighted_p:>12.4f} {weighted_r:>12.4f} {weighted_f1:>12.4f} {total:>10}")
    
    lines.append("")
    lines.append(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    return "\n".join(lines)
