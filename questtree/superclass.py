"""
Super-Class Clustering (Algorithm 4) for QUEST Decision Tree.

Converts K-class problems to binary by clustering class centroids
into two super-classes using 2-means clustering.

References
----------
Loh, W.-Y., & Shih, Y.-S. (1997). Split selection methods for classification trees.
Statistica Sinica, 7, 815-840.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Set, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2

from .feature_detection import is_categorical


def compute_superclass_labels(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: Optional[List[str]] = None
) -> Tuple[np.ndarray, Set, Set]:
    """
    Apply Super-Class Clustering (Algorithm 4) for multiclass problems.
    
    QUEST converts K-class problems to binary by clustering class centroids
    into two super-classes. This preserves the binary QDA/CRIMCOORDS machinery
    while properly handling all K classes.
    
    Mathematical Formulation
    ------------------------
    1. **Compute Class Centroids**:
       For each class k ∈ {1, ..., K}:
       
       μₖ = (1/nₖ) Σᵢ:yᵢ=k xᵢ
       
       where nₖ is the number of samples in class k.
    
    2. **Apply 2-Means Clustering**:
       Cluster the K centroids {μ₁, ..., μₖ} into two groups:
       
       min_{C₀,C₁} Σₖ∈C₀ ||μₖ - c₀||² + Σₖ∈C₁ ||μₖ - c₁||²
       
       where c₀, c₁ are the cluster centers.
    
    3. **Partition Classes**:
       - Super-class A = {k : centroid k assigned to cluster 0}
       - Super-class B = {k : centroid k assigned to cluster 1}
    
    4. **Create Binary Labels**:
       For each sample i:
       ỹᵢ = 0 if yᵢ ∈ A, else 1
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray of shape (n_samples,)
        Original class labels (can have K ≥ 2 unique values).
    feature_types : list of str, optional
        'continuous' or 'categorical' for each feature.
        If None, types are inferred.
    
    Returns
    -------
    y_binary : np.ndarray of shape (n_samples,)
        Binary labels (0 or 1) for each sample.
    group_A : set
        Set of original class labels assigned to super-class 0.
    group_B : set
        Set of original class labels assigned to super-class 1.
    
    Notes
    -----
    - For K=2 (already binary), this is essentially a no-op that maps
      the first class to 0 and second to 1.
    - For all-categorical data, classes are split by sample counts to
      balance the binary problem.
    - The 2-means clustering is run multiple times with different seeds
      to find the best clustering.
    
    Examples
    --------
    >>> X = np.random.randn(150, 4)
    >>> y = np.array([0]*50 + [1]*50 + [2]*50)  # 3 classes
    >>> y_binary, group_A, group_B = compute_superclass_labels(X, y)
    >>> len(group_A) + len(group_B) == 3
    True
    >>> set(y_binary) == {0, 1}
    True
    """
    classes = np.unique(y)
    K = len(classes)
    
    # =========================================================================
    # Trivial Case: Already Binary (K = 2)
    # =========================================================================
    if K == 2:
        label_map = {classes[0]: 0, classes[1]: 1}
        y_binary = np.array([label_map[label] for label in y])
        return y_binary, {classes[0]}, {classes[1]}
    
    # =========================================================================
    # Edge Case: Single Class (K = 1)
    # =========================================================================
    if K == 1:
        return np.zeros(len(y), dtype=int), set(classes), set()
    
    n_features = X.shape[1]
    
    # Infer feature types if not provided
    if feature_types is None:
        feature_types = []
        for j in range(n_features):
            if is_categorical(X[:, j]):
                feature_types.append('categorical')
            else:
                feature_types.append('continuous')
    
    # Identify continuous features for centroid computation
    continuous_indices = [j for j in range(n_features) 
                         if feature_types[j] == 'continuous']
    
    # =========================================================================
    # Special Case: All Categorical Features
    # =========================================================================
    if len(continuous_indices) == 0:
        # Split classes by sample counts to balance groups
        class_counts = [(c, np.sum(y == c)) for c in classes]
        class_counts.sort(key=lambda x: x[1], reverse=True)
        
        total = sum(count for _, count in class_counts)
        cumsum = 0
        group_A: Set = set()
        group_B: Set = set()
        
        for c, count in class_counts:
            if cumsum < total / 2:
                group_A.add(c)
                cumsum += count
            else:
                group_B.add(c)
        
        # Ensure both groups are non-empty
        if len(group_B) == 0:
            group_B.add(group_A.pop())
        
        y_binary = np.array([0 if label in group_A else 1 for label in y])
        return y_binary, group_A, group_B
    
    # =========================================================================
    # Step 1: Compute Class Centroids
    # =========================================================================
    # Use only continuous features for distance computation
    centroids = np.zeros((K, len(continuous_indices)), dtype=np.float64)
    
    for k_idx, c in enumerate(classes):
        mask = (y == c)
        class_data = X[mask][:, continuous_indices].astype(np.float64)
        centroids[k_idx] = np.mean(class_data, axis=0)
    
    # =========================================================================
    # Step 2: Apply 2-Means Clustering to Centroids
    # =========================================================================
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            best_labels = None
            best_inertia = np.inf
            
            # Run multiple times with different seeds
            for seed in range(5):
                np.random.seed(seed)
                try:
                    _, labels = kmeans2(centroids, 2, iter=20, minit='points')
                    
                    # Compute inertia (sum of squared distances)
                    cluster_centers = np.array([
                        np.mean(centroids[labels == 0], axis=0) if np.any(labels == 0) else centroids[0],
                        np.mean(centroids[labels == 1], axis=0) if np.any(labels == 1) else centroids[-1]
                    ])
                    
                    inertia = 0.0
                    for i, lab in enumerate(labels):
                        inertia += np.sum((centroids[i] - cluster_centers[lab])**2)
                    
                    if inertia < best_inertia:
                        best_inertia = inertia
                        best_labels = labels
                        
                except Exception:
                    continue
            
            # Fallback: Use PCA-based split
            if best_labels is None:
                pca_direction = np.mean(centroids, axis=0)
                projections = centroids @ pca_direction
                median_proj = np.median(projections)
                best_labels = (projections > median_proj).astype(int)
            
            labels = best_labels
            
    except Exception:
        # Ultimate fallback: split classes in half
        labels = np.zeros(K, dtype=int)
        labels[K // 2:] = 1
    
    # =========================================================================
    # Step 3: Build Super-Class Groups
    # =========================================================================
    group_A = set(classes[labels == 0])
    group_B = set(classes[labels == 1])
    
    # Ensure both groups are non-empty
    if len(group_A) == 0:
        group_A.add(group_B.pop())
    if len(group_B) == 0:
        group_B.add(group_A.pop())
    
    # =========================================================================
    # Step 4: Create Binary Labels
    # =========================================================================
    y_binary = np.array([0 if label in group_A else 1 for label in y])
    
    return y_binary, group_A, group_B


def get_class_centroids(
    X: np.ndarray,
    y: np.ndarray,
    feature_types: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Compute class centroids using continuous features only.
    
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
    centroids : np.ndarray of shape (n_classes, n_continuous_features)
        Class centroids.
    classes : np.ndarray of shape (n_classes,)
        Unique class labels.
    continuous_indices : list of int
        Indices of continuous features used.
    """
    classes = np.unique(y)
    K = len(classes)
    n_features = X.shape[1]
    
    if feature_types is None:
        feature_types = []
        for j in range(n_features):
            if is_categorical(X[:, j]):
                feature_types.append('categorical')
            else:
                feature_types.append('continuous')
    
    continuous_indices = [j for j in range(n_features) 
                         if feature_types[j] == 'continuous']
    
    if len(continuous_indices) == 0:
        return np.zeros((K, 0)), classes, []
    
    centroids = np.zeros((K, len(continuous_indices)), dtype=np.float64)
    
    for k_idx, c in enumerate(classes):
        mask = (y == c)
        class_data = X[mask][:, continuous_indices].astype(np.float64)
        centroids[k_idx] = np.mean(class_data, axis=0)
    
    return centroids, classes, continuous_indices
