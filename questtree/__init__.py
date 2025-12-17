"""
QUEST Decision Tree Classifier.

Quick, Unbiased, Efficient Statistical Tree - A decision tree algorithm
that achieves unbiased variable selection through statistical hypothesis
tests and finds optimal split points using Quadratic Discriminant Analysis.

This implementation follows the original QUEST paper:
    Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for
    classification trees. Statistica Sinica, 7, 815-840.

Basic Usage
-----------
>>> from questtree import QuestTreeClassifier
>>> 
>>> # Create and fit classifier (automatic feature type detection)
>>> clf = QuestTreeClassifier()
>>> clf.fit(X_train, y_train)
>>>
>>> # Predict
>>> y_pred = clf.predict(X_test)
>>> accuracy = clf.score(X_test, y_test)

With Pruning
------------
>>> clf = QuestTreeClassifier(ccp_alpha=0.01)
>>> clf.fit(X_train, y_train)

With Manual Feature Types
-------------------------
>>> clf = QuestTreeClassifier(
...     feature_types=['continuous', 'continuous', 'categorical']
... )
>>> clf.fit(X, y)

Visualization
-------------
>>> from questtree import print_tree, get_tree_summary
>>> print_tree(clf)
>>> summary = get_tree_summary(clf)

Key Features
------------
- Unbiased variable selection via ANOVA/Chi-square tests
- Multiclass classification via super-class clustering
- Categorical variable handling via CRIMCOORDS
- Cost-complexity pruning
- Automatic feature type detection
- Scikit-learn compatible API (without sklearn dependency)

Module Structure
----------------
questtree/
├── __init__.py         # This file - main exports
├── classifier.py       # QuestTreeClassifier
├── node.py             # Node data structure
├── splits.py           # QDA split & CRIMCOORDS (Algorithms 1 & 2)
├── variable_selection.py  # Statistical tests (Algorithm 3)
├── superclass.py       # Super-class clustering (Algorithm 4)
├── pruning.py          # Cost-complexity pruning
├── feature_detection.py   # Automatic feature type detection
├── validation.py       # Input validation
├── plotting.py         # Visualization utilities
└── utils.py            # Helper functions

Dependencies
------------
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas (optional, for DataFrame support)

No scikit-learn required.
"""

__version__ = "1.0.0"
__author__ = "QUEST Implementation"

# =============================================================================
# Main Classifier
# =============================================================================
from .classifier import QuestTreeClassifier

# =============================================================================
# Visualization
# =============================================================================
from .plotting import (
    print_tree,
    get_tree_summary,
    get_feature_importances,
    tree_to_rules,
    export_text,
    plot_pruning_path
)

# =============================================================================
# Feature Detection
# =============================================================================
from .feature_detection import (
    FeatureTypeDetector,
    infer_feature_types,
    is_categorical
)

# =============================================================================
# Utilities
# =============================================================================
from .utils import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =============================================================================
# Advanced/Internal (for users who want fine-grained control)
# =============================================================================
from .node import Node
from .validation import ValidationError

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Main classifier
    "QuestTreeClassifier",
    
    # Visualization
    "print_tree",
    "get_tree_summary",
    "get_feature_importances",
    "tree_to_rules",
    "export_text",
    "plot_pruning_path",
    
    # Feature detection
    "FeatureTypeDetector",
    "infer_feature_types",
    "is_categorical",
    
    # Utilities
    "accuracy_score",
    "confusion_matrix",
    "classification_report",
    
    # Advanced
    "Node",
    "ValidationError",
]


# =============================================================================
# Module-level convenience functions
# =============================================================================

def __getattr__(name: str):
    """Lazy loading for less commonly used submodules."""
    if name == "splits":
        from . import splits
        return splits
    elif name == "variable_selection":
        from . import variable_selection
        return variable_selection
    elif name == "superclass":
        from . import superclass
        return superclass
    elif name == "pruning":
        from . import pruning
        return pruning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
