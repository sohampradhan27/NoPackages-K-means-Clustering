"""
Unit tests for K-means clustering implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kmeans import KMeans
from examples.sample_data_generator import generate_sample_data


def test_kmeans_fit():
    """Test that K-means can fit to data without errors."""
    print("Testing K-means fit...")
    
    # Generate test data
    data = generate_sample_data(n_samples=50, n_clusters=2, random_state=42)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(data)
    
    # Check that centroids and labels are set
    assert kmeans.centroids is not None, "Centroids not set after fit"
    assert kmeans.labels is not None, "Labels not set after fit"
    assert len(kmeans.centroids) == 2, "Wrong number of centroids"
    assert len(kmeans.labels) == 50, "Wrong number of labels"
    
    print("✓ K-means fit test passed")


def test_kmeans_predict():
    """Test prediction on new data."""
    print("Testing K-means predict...")
    
    # Train on one dataset
    train_data = generate_sample_data(n_samples=50, n_clusters=2, random_state=42)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(train_data)
    
    # Predict on new data
    test_data = generate_sample_data(n_samples=20, n_clusters=2, random_state=123)
    predictions = kmeans.predict(test_data)
    
    assert len(predictions) == 20, "Wrong number of predictions"
    assert all(0 <= label < 2 for label in predictions), "Invalid cluster labels"
    
    print("✓ K-means predict test passed")


def test_convergence():
    """Test that K-means converges on simple data."""
    print("Testing K-means convergence...")
    
    # Create clearly separated clusters
    data = [
        [1, 1], [1, 2], [2, 1], [2, 2],  # Cluster 0
        [8, 8], [8, 9], [9, 8], [9, 9]   # Cluster 1
    ]
    
    kmeans = KMeans(n_clusters=2, max_iters=100, tolerance=1e-4)
    kmeans.fit(data)
    
    # Check convergence
    assert kmeans.n_iterations < 100, "Failed to converge"
    
    # Check that clusters are correctly identified
    cluster_0_labels = kmeans.labels[:4]
    cluster_1_labels = kmeans.labels[4:]
    
    assert len(set(cluster_0_labels)) == 1, "First cluster not uniform"
    assert len(set(cluster_1_labels)) == 1, "Second cluster not uniform"
    assert set(cluster_0_labels) != set(cluster_1_labels), "Clusters not separated"
    
    print(f"✓ K-means convergence test passed (converged in {kmeans.n_iterations} iterations)")


def test_inertia():
    """Test inertia calculation."""
    print("Testing inertia calculation...")
    
    # Simple data where we can calculate inertia manually
    data = [[0, 0], [2, 0], [0, 2], [2, 2]]
    
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(data)
    
    inertia = kmeans.inertia(data)
    assert inertia > 0, "Inertia should be positive"
    
    # With more clusters, inertia should decrease
    kmeans2 = KMeans(n_clusters=2, random_state=42)
    kmeans2.fit(data)
    inertia2 = kmeans2.inertia(data)
    
    assert inertia2 < inertia, "Inertia should decrease with more clusters"
    
    print("✓ Inertia test passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    kmeans = KMeans(n_clusters=2)
    
    # Test empty data
    try:
        kmeans.fit([])
        assert False, "Should raise error for empty data"
    except ValueError:
        pass
    
    # Test too few samples
    try:
        kmeans.fit([[1, 1]])
        assert False, "Should raise error for too few samples"
    except ValueError:
        pass
    
    # Test predict before fit
    try:
        new_kmeans = KMeans(n_clusters=2)
        new_kmeans.predict([[1, 1]])
        assert False, "Should raise error for predict before fit"
    except ValueError:
        pass
    
    print("✓ Edge cases test passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*50)
    print("RUNNING K-MEANS UNIT TESTS")
    print("="*50 + "\n")
    
    test_kmeans_fit()
    test_kmeans_predict()
    test_convergence()
    test_inertia()
    test_edge_cases()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED! ✓")
    print("="*50)


if __name__ == "__main__":
    run_all_tests()
