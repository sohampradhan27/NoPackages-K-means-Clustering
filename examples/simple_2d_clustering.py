#!/usr/bin/env python3
"""
Simple example demonstrating K-means clustering on 2D data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kmeans import KMeans
from src.visualization import plot_clusters, print_cluster_statistics
from examples.sample_data_generator import (
    generate_sample_data, 
    generate_moon_data, 
    generate_ring_data
)


def run_clustering_example(data, n_clusters, dataset_name):
    """Run K-means clustering on given data and display results."""
    print(f"\n{'='*60}")
    print(f"K-MEANS CLUSTERING: {dataset_name}")
    print('='*60)
    
    # Initialize and run K-means
    kmeans = KMeans(n_clusters=n_clusters, max_iters=100, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # Print convergence info
    print(f"\nConverged in {kmeans.n_iterations} iterations")
    print(f"Final inertia: {kmeans.inertia(data):.2f}")
    
    # Visualize results
    plot_clusters(data, labels, kmeans.centroids)
    
    # Print statistics
    print_cluster_statistics(data, labels, kmeans.centroids)


def main():
    """Main function to run various clustering examples."""
    
    print("K-MEANS CLUSTERING DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows K-means clustering on different datasets.")
    print("All visualizations use ASCII characters.")
    
    # Example 1: Well-separated Gaussian clusters
    print("\n" + "─"*60)
    print("Example 1: Well-separated Gaussian clusters")
    print("─"*60)
    data1 = generate_sample_data(n_samples=120, n_clusters=3, 
                                 cluster_spread=0.5, random_state=42)
    run_clustering_example(data1, n_clusters=3, 
                          dataset_name="Gaussian Clusters")
    
    # Example 2: Moon-shaped data
    print("\n" + "─"*60)
    print("Example 2: Moon-shaped data")
    print("─"*60)
    print("Note: K-means may struggle with non-spherical clusters")
    data2 = generate_moon_data(n_samples=100, noise=0.05, random_state=42)
    run_clustering_example(data2, n_clusters=2, 
                          dataset_name="Moon Shapes")
    
    # Example 3: Ring data
    print("\n" + "─"*60)
    print("Example 3: Concentric rings")
    print("─"*60)
    print("Note: K-means assumes spherical clusters")
    data3 = generate_ring_data(n_samples=90, random_state=42)
    run_clustering_example(data3, n_clusters=3, 
                          dataset_name="Concentric Rings")
    
    # Interactive example
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Generate new random data")
        print("2. Try different number of clusters on last dataset")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            n_samples = int(input("Number of samples (e.g., 100): "))
            n_clusters = int(input("Number of true clusters (e.g., 3): "))
            k = int(input("Number of clusters to find (e.g., 3): "))
            
            data = generate_sample_data(n_samples=n_samples, 
                                      n_clusters=n_clusters,
                                      random_state=None)
            run_clustering_example(data, n_clusters=k, 
                                 dataset_name="Custom Dataset")
        
        elif choice == "2":
            if 'data' not in locals():
                print("No dataset available. Generate one first.")
                continue
            
            k = int(input("Number of clusters to find: "))
            run_clustering_example(data, n_clusters=k, 
                                 dataset_name="Custom Dataset")
        
        elif choice == "3":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
