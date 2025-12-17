"""
Example Usage for QUEST Decision Tree Classifier.

This module provides example code demonstrating how to use the
QuestTreeClassifier with various datasets and configurations.

Run this file directly to see the examples in action:
    python -m questtree.examples
"""

import numpy as np


def example_basic_usage():
    """
    Basic Usage Example.
    
    Shows the simplest way to use QuestTreeClassifier with
    automatic feature type detection.
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    from questtree import QuestTreeClassifier
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    
    # Two informative features, one noise feature
    X = np.column_stack([
        np.concatenate([np.random.normal(2, 1, 100), np.random.normal(5, 1, 100)]),
        np.concatenate([np.random.normal(0, 0.5, 100), np.random.normal(0, 2, 100)]),
        np.random.randn(n_samples)  # noise
    ])
    y = np.array([0] * 100 + [1] * 100)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    # Train-test split
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Create and fit classifier (automatic feature detection)
    clf = QuestTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    
    print(f"\nClassifier: {clf}")
    print(f"Feature types detected: {clf.feature_types_}")
    print(f"Number of leaves: {clf.get_n_leaves()}")
    print(f"Tree depth: {clf.get_depth()}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    return clf


def example_with_pruning():
    """
    Example with Cost-Complexity Pruning.
    
    Demonstrates how pruning reduces tree complexity while
    maintaining good accuracy.
    """
    print("\n" + "=" * 60)
    print("Example 2: With Cost-Complexity Pruning")
    print("=" * 60)
    
    from questtree import QuestTreeClassifier
    
    # Generate data
    np.random.seed(42)
    n_samples = 300
    
    X = np.random.randn(n_samples, 5)
    # Create classes based on first two features
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    y[(X[:, 0] < 0) & (X[:, 1] < 0)] = 2  # Third class
    
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    
    # Train with different alpha values
    print("\nPruning comparison:")
    print("-" * 50)
    
    for alpha in [0.0, 0.01, 0.02, 0.05]:
        clf = QuestTreeClassifier(ccp_alpha=alpha, random_state=42)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print(f"α = {alpha:.2f}: leaves = {clf.get_n_leaves():2d}, "
              f"train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f}")
    
    return clf


def example_categorical_features():
    """
    Example with Categorical Features.
    
    Demonstrates automatic detection and CRIMCOORDS transformation
    of categorical variables.
    """
    print("\n" + "=" * 60)
    print("Example 3: Categorical Features")
    print("=" * 60)
    
    from questtree import QuestTreeClassifier
    
    # Generate mixed data
    np.random.seed(42)
    n_samples = 200
    
    # Continuous feature
    feat_continuous = np.random.randn(n_samples)
    
    # Categorical feature correlated with class
    categories = np.array(['A', 'B', 'C', 'D'])
    feat_categorical = np.concatenate([
        np.random.choice(['A', 'B'], 100, p=[0.7, 0.3]),  # Class 0: mostly A, B
        np.random.choice(['C', 'D'], 100, p=[0.7, 0.3])   # Class 1: mostly C, D
    ])
    
    # Create feature matrix (must use object dtype for mixed types)
    X = np.column_stack([feat_continuous, feat_categorical])
    y = np.array([0] * 100 + [1] * 100)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    # Fit classifier (auto-detects categorical feature)
    clf = QuestTreeClassifier()
    clf.fit(X_train, y_train)
    
    print(f"\nFeature types: {clf.feature_types_}")
    print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")
    
    return clf


def example_multiclass():
    """
    Example with Multiclass Classification.
    
    Demonstrates Super-Class Clustering for K > 2 classes.
    """
    print("\n" + "=" * 60)
    print("Example 4: Multiclass Classification (K=4)")
    print("=" * 60)
    
    from questtree import QuestTreeClassifier, print_tree
    
    # Generate 4-class data
    np.random.seed(42)
    
    # 4 class centers
    centers = np.array([[0, 0], [0, 3], [3, 0], [3, 3]])
    
    X = []
    y = []
    for i, center in enumerate(centers):
        X.append(np.random.randn(50, 2) + center)
        y.append(np.full(50, i))
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    # Fit classifier
    clf = QuestTreeClassifier(ccp_alpha=0.01, random_state=42)
    clf.fit(X_train, y_train)
    
    print(f"\nClasses: {list(clf.classes_)}")
    print(f"Number of leaves: {clf.get_n_leaves()}")
    print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")
    
    # Show tree structure
    print("\nTree structure:")
    print_tree(clf)
    
    return clf


def example_sklearn_api():
    """
    Example demonstrating sklearn-like API.
    
    Shows get_params, set_params, predict_proba, etc.
    """
    print("\n" + "=" * 60)
    print("Example 5: Scikit-learn Compatible API")
    print("=" * 60)
    
    from questtree import QuestTreeClassifier, classification_report
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    # Create classifier
    clf = QuestTreeClassifier(max_depth=5, ccp_alpha=0.02)
    
    # get_params
    print("\n1. get_params():")
    print(f"   {clf.get_params()}")
    
    # set_params
    clf.set_params(max_depth=10, ccp_alpha=0.01)
    print("\n2. After set_params(max_depth=10, ccp_alpha=0.01):")
    print(f"   {clf.get_params()}")
    
    # fit
    clf.fit(X_train, y_train)
    
    # predict
    y_pred = clf.predict(X_test)
    print(f"\n3. predict() sample: {y_pred[:10]}")
    
    # predict_proba
    proba = clf.predict_proba(X_test)
    print(f"\n4. predict_proba() sample:\n{proba[:5]}")
    
    # score
    print(f"\n5. score(): {clf.score(X_test, y_test):.4f}")
    
    # Classification report (our own implementation, no sklearn)
    print("\n6. Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf


def example_feature_importances():
    """
    Example showing feature importance computation.
    """
    print("\n" + "=" * 60)
    print("Example 6: Feature Importances")
    print("=" * 60)
    
    from questtree import QuestTreeClassifier, get_feature_importances
    
    # Generate data with known important features
    np.random.seed(42)
    n_samples = 300
    
    # Feature 0: Very informative
    # Feature 1: Moderately informative
    # Features 2-4: Noise
    X = np.column_stack([
        np.concatenate([np.random.normal(0, 1, 150), np.random.normal(3, 1, 150)]),  # Important
        np.concatenate([np.random.normal(0, 1, 150), np.random.normal(1, 1, 150)]),  # Less important
        np.random.randn(n_samples),  # Noise
        np.random.randn(n_samples),  # Noise
        np.random.randn(n_samples),  # Noise
    ])
    y = np.array([0] * 150 + [1] * 150)
    
    idx = np.random.permutation(n_samples)
    X, y = X[idx], y[idx]
    
    # Fit
    clf = QuestTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    # Get importances
    feature_names = ['important', 'moderate', 'noise_1', 'noise_2', 'noise_3']
    importances = get_feature_importances(clf, feature_names)
    
    print("\nFeature Importances (by split frequency):")
    print("-" * 40)
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"{name:12s}: {imp:.4f} {bar}")
    
    return clf


def run_all_examples():
    """Run all examples."""
    example_basic_usage()
    example_with_pruning()
    example_categorical_features()
    example_multiclass()
    example_sklearn_api()
    example_feature_importances()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
