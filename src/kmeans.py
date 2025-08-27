"""
K-means clustering implementation using only Python standard library.
"""

import random
import math
from typing import List, Tuple, Optional


class KMeans:
    """
    K-means clustering algorithm implementation.
    
    Attributes:
        n_clusters: Number of clusters to form
        max_iters: Maximum number of iterations
        tolerance: Convergence tolerance for centroid changes
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, 
                 tolerance: float = 1e-4, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.n_iterations = 0
        
        if random_state is not None:
            random.seed(random_state)
    
    def _euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point coordinates
            point2: Second point coordinates
            
        Returns:
            Euclidean distance between the points
        """
        if len(point1) != len(point2):
            raise ValueError("Points must have the same dimension")
        
        squared_sum = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
        return math.sqrt(squared_sum)
    
    def _initialize_centroids(self, data: List[List[float]]) -> List[List[float]]:
        """
        Initialize centroids using random data points (Forgy method).
        
        Args:
            data: Input data points
            
        Returns:
            Initial centroids
        """
        # Randomly select k data points as initial centroids
        indices = random.sample(range(len(data)), self.n_clusters)
        centroids = [data[i].copy() for i in indices]
        return centroids
    
    def _assign_clusters(self, data: List[List[float]], 
                        centroids: List[List[float]]) -> List[int]:
        """
        Assign each data point to the nearest centroid.
        
        Args:
            data: Input data points
            centroids: Current centroid positions
            
        Returns:
            Cluster labels for each data point
        """
        labels = []
        
        for point in data:
            # Calculate distances to all centroids
            distances = [self._euclidean_distance(point, centroid) 
                        for centroid in centroids]
            
            # Assign to nearest centroid
            nearest_cluster = distances.index(min(distances))
            labels.append(nearest_cluster)
        
        return labels
    
    def _update_centroids(self, data: List[List[float]], 
                         labels: List[int]) -> List[List[float]]:
        """
        Update centroids as the mean of assigned points.
        
        Args:
            data: Input data points
            labels: Current cluster assignments
            
        Returns:
            Updated centroids
        """
        n_features = len(data[0]) if data else 0
        new_centroids = []
        
        for cluster_id in range(self.n_clusters):
            # Get all points assigned to this cluster
            cluster_points = [data[i] for i, label in enumerate(labels) 
                             if label == cluster_id]
            
            if not cluster_points:
                # If cluster has no points, keep the old centroid
                new_centroids.append(self.centroids[cluster_id].copy())
            else:
                # Calculate mean of all points in the cluster
                centroid = []
                for feature_idx in range(n_features):
                    mean_value = sum(point[feature_idx] for point in cluster_points) / len(cluster_points)
                    centroid.append(mean_value)
                new_centroids.append(centroid)
        
        return new_centroids
    
    def _has_converged(self, old_centroids: List[List[float]], 
                      new_centroids: List[List[float]]) -> bool:
        """
        Check if centroids have converged (moved less than tolerance).
        
        Args:
            old_centroids: Previous centroid positions
            new_centroids: Current centroid positions
            
        Returns:
            True if converged, False otherwise
        """
        for old, new in zip(old_centroids, new_centroids):
            distance = self._euclidean_distance(old, new)
            if distance > self.tolerance:
                return False
        return True
    
    def fit(self, data: List[List[float]]) -> 'KMeans':
        """
        Fit K-means clustering to data.
        
        Args:
            data: Input data points (list of lists)
            
        Returns:
            Self (fitted KMeans object)
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        if len(data) < self.n_clusters:
            raise ValueError(f"Number of samples ({len(data)}) must be >= n_clusters ({self.n_clusters})")
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(data)
        
        # Iterative optimization
        for iteration in range(self.max_iters):
            # Step 1: Assign points to nearest centroid
            self.labels = self._assign_clusters(data, self.centroids)
            
            # Step 2: Update centroids
            new_centroids = self._update_centroids(data, self.labels)
            
            # Step 3: Check for convergence
            if self._has_converged(self.centroids, new_centroids):
                self.n_iterations = iteration + 1
                break
            
            self.centroids = new_centroids
        else:
            self.n_iterations = self.max_iters
        
        return self
    
    def predict(self, data: List[List[float]]) -> List[int]:
        """
        Predict cluster labels for new data points.
        
        Args:
            data: Data points to predict
            
        Returns:
            Cluster labels
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self._assign_clusters(data, self.centroids)
    
    def fit_predict(self, data: List[List[float]]) -> List[int]:
        """
        Fit the model and return cluster labels.
        
        Args:
            data: Input data points
            
        Returns:
            Cluster labels
        """
        self.fit(data)
        return self.labels
    
    def inertia(self, data: List[List[float]]) -> float:
        """
        Calculate sum of squared distances to nearest centroid (inertia).
        
        Args:
            data: Input data points
            
        Returns:
            Inertia value
        """
        if self.centroids is None or self.labels is None:
            raise ValueError("Model must be fitted first")
        
        total_distance = 0
        for i, point in enumerate(data):
            centroid = self.centroids[self.labels[i]]
            total_distance += self._euclidean_distance(point, centroid) ** 2
        
        return total_distance
