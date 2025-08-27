"""
Sample data generation utilities for testing K-means clustering.
"""

import random
import math
from typing import List, Tuple


def generate_gaussian_cluster(center: List[float], spread: float, 
                             n_points: int) -> List[List[float]]:
    """
    Generate points from a Gaussian distribution around a center.
    
    Args:
        center: Center coordinates of the cluster
        spread: Standard deviation of the distribution
        n_points: Number of points to generate
        
    Returns:
        List of generated points
    """
    points = []
    for _ in range(n_points):
        point = []
        for coord in center:
            # Box-Muller transform for normal distribution
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            value = coord + z * spread
            point.append(value)
        points.append(point)
    return points


def generate_sample_data(n_samples: int = 150, n_clusters: int = 3, 
                        n_features: int = 2, cluster_spread: float = 1.0,
                        random_state: int = None) -> List[List[float]]:
    """
    Generate synthetic clustered data for testing.
    
    Args:
        n_samples: Total number of samples to generate
        n_clusters: Number of clusters
        n_features: Number of features (dimensions)
        cluster_spread: Standard deviation of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        List of data points
    """
    if random_state is not None:
        random.seed(random_state)
    
    # Generate cluster centers
    centers = []
    for _ in range(n_clusters):
        center = [random.uniform(-10, 10) for _ in range(n_features)]
        centers.append(center)
    
    # Generate points for each cluster
    data = []
    points_per_cluster = n_samples // n_clusters
    remaining = n_samples % n_clusters
    
    for i, center in enumerate(centers):
        # Add extra point to first clusters if n_samples not divisible by n_clusters
        n_points = points_per_cluster + (1 if i < remaining else 0)
        cluster_points = generate_gaussian_cluster(center, cluster_spread, n_points)
        data.extend(cluster_points)
    
    # Shuffle the data
    random.shuffle(data)
    
    return data


def generate_moon_data(n_samples: int = 200, noise: float = 0.1,
                       random_state: int = None) -> List[List[float]]:
    """
    Generate two interleaving half-moon shapes.
    
    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        random_state: Random seed
        
    Returns:
        List of 2D data points
    """
    if random_state is not None:
        random.seed(random_state)
    
    n_samples_per_moon = n_samples // 2
    data = []
    
    # First moon
    for i in range(n_samples_per_moon):
        angle = math.pi * i / n_samples_per_moon
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
    
    # Second moon (shifted and flipped)
    for i in range(n_samples_per_moon):
        angle = math.pi * i / n_samples_per_moon
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 0.5 - math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
    
    random.shuffle(data)
    return data


def generate_ring_data(n_samples: int = 150, random_state: int = None) -> List[List[float]]:
    """
    Generate data points in concentric rings.
    
    Args:
        n_samples: Total number of samples
        random_state: Random seed
        
    Returns:
        List of 2D data points
    """
    if random_state is not None:
        random.seed(random_state)
    
    data = []
    n_rings = 3
    samples_per_ring = n_samples // n_rings
    
    for ring in range(n_rings):
        radius = (ring + 1) * 2
        for _ in range(samples_per_ring):
            angle = random.uniform(0, 2 * math.pi)
            noise = random.gauss(0, 0.1)
            x = (radius + noise) * math.cos(angle)
            y = (radius + noise) * math.sin(angle)
            data.append([x, y])
    
    return data
