"""
ASCII-based visualization for K-means clustering results.
"""

from typing import List, Optional, Tuple
import math


def plot_clusters(data: List[List[float]], labels: List[int], 
                 centroids: Optional[List[List[float]]] = None,
                 width: int = 60, height: int = 20) -> None:
    """
    Create an ASCII plot of clustered data points.
    
    Args:
        data: Data points (2D only)
        labels: Cluster labels for each point
        centroids: Cluster centroids (optional)
        width: Plot width in characters
        height: Plot height in characters
    """
    if not data or not all(len(point) == 2 for point in data):
        print("Error: Can only plot 2D data")
        return
    
    # Get data bounds
    x_coords = [point[0] for point in data]
    y_coords = [point[1] for point in data]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Create empty plot
    plot = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Define cluster symbols
    symbols = ['●', '▲', '■', '♦', '★', '▼', '◆', '○', '△', '□']
    
    # Plot data points
    for point, label in zip(data, labels):
        x, y = point[0], point[1]
        
        # Map to plot coordinates
        plot_x = int((x - x_min) / (x_max - x_min) * (width - 1))
        plot_y = int((1 - (y - y_min) / (y_max - y_min)) * (height - 1))
        
        # Ensure within bounds
        plot_x = max(0, min(width - 1, plot_x))
        plot_y = max(0, min(height - 1, plot_y))
        
        # Place symbol
        symbol = symbols[label % len(symbols)]
        plot[plot_y][plot_x] = symbol
    
    # Plot centroids if provided
    if centroids:
        for i, centroid in enumerate(centroids):
            x, y = centroid[0], centroid[1]
            
            # Map to plot coordinates
            plot_x = int((x - x_min) / (x_max - x_min) * (width - 1))
            plot_y = int((1 - (y - y_min) / (y_max - y_min)) * (height - 1))
            
            # Ensure within bounds
            plot_x = max(0, min(width - 1, plot_x))
            plot_y = max(0, min(height - 1, plot_y))
            
            # Place centroid marker
            plot[plot_y][plot_x] = str(i)
    
    # Print plot with border
    print("\n" + "─" * (width + 2))
    for row in plot:
        print("│" + "".join(row) + "│")
    print("─" * (width + 2))
    
    # Print legend
    print("\nLegend:")
    n_clusters = len(set(labels))
    for i in range(n_clusters):
        symbol = symbols[i % len(symbols)]
        print(f"  Cluster {i}: {symbol}")
    if centroids:
        print("  Centroids: 0,1,2,...")
    
    # Print axis info
    print(f"\nX-axis: [{x_min:.2f}, {x_max:.2f}]")
    print(f"Y-axis: [{y_min:.2f}, {y_max:.2f}]")


def print_cluster_statistics(data: List[List[float]], labels: List[int], 
                            centroids: List[List[float]]) -> None:
    """
    Print statistics about the clustering results.
    
    Args:
        data: Data points
        labels: Cluster labels
        centroids: Cluster centroids
    """
    n_clusters = len(centroids)
    
    print("\n" + "="*50)
    print("CLUSTERING STATISTICS")
    print("="*50)
    
    for cluster_id in range(n_clusters):
        cluster_points = [data[i] for i, label in enumerate(labels) 
                         if label == cluster_id]
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {len(cluster_points)} points")
        print(f"  Centroid: {[f'{x:.2f}' for x in centroids[cluster_id]]}")
        
        if cluster_points:
            # Calculate cluster spread (average distance from centroid)
            distances = []
            for point in cluster_points:
                dist = math.sqrt(sum((p - c)**2 for p, c in 
                               zip(point, centroids[cluster_id])))
                distances.append(dist)
            
            avg_distance = sum(distances) / len(distances)
            max_distance = max(distances)
            min_distance = min(distances)
            
            print(f"  Average distance from centroid: {avg_distance:.2f}")
            print(f"  Max distance from centroid: {max_distance:.2f}")
            print(f"  Min distance from centroid: {min_distance:.2f}")
