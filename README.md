# K-means Clustering from Scratch 🎯

A pure Python implementation of K-means clustering algorithm with **zero external dependencies**. Features ASCII-based visualization and comprehensive educational documentation.

![Python](https://img.shields.io/badge/python-3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-none-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **Zero Dependencies**: Uses only Python standard library - no NumPy, SciPy, or Matplotlib required
- **ASCII Visualization**: Terminal-based plotting that works everywhere
- **Educational Focus**: Extensive comments and clear implementation
- **Interactive Mode**: Experiment with different parameters in real-time
- **Multiple Data Generators**: Test with various cluster shapes and distributions
- **Comprehensive Testing**: Unit tests ensure correctness

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kmeans-from-scratch.git
cd kmeans-from-scratch

# No installation required! Just run:
python examples/simple_2d_clustering.py
```

### Basic Usage

```python
from src.kmeans import KMeans
from examples.sample_data_generator import generate_sample_data

# Generate sample data
data = generate_sample_data(n_samples=100, n_clusters=3)

# Cluster the data
kmeans = KMeans(n_clusters=3, max_iters=100)
labels = kmeans.fit_predict(data)

# Get cluster centers
centroids = kmeans.centroids
```

## 📊 Sample Output

Running the example script produces beautiful ASCII visualizations:

```
============================================================
K-MEANS CLUSTERING: Gaussian Clusters
============================================================

Converged in 5 iterations
Final inertia: 142.38

──────────────────────────────────────────────────────────────
│                              ●●●                            │
│                            ●●●●●●●                          │
│                           ●●●0●●●●                          │
│                            ●●●●●●                           │
│                              ●●                             │
│                                                              │
│      ▲▲▲▲                                    ■■■            │
│    ▲▲▲▲▲▲▲                                 ■■■■■■           │
│    ▲▲▲1▲▲▲                                ■■■2■■■■          │
│     ▲▲▲▲▲                                  ■■■■■■           │
│       ▲▲                                     ■■             │
──────────────────────────────────────────────────────────────

Legend:
  Cluster 0: ●
  Cluster 1: ▲
  Cluster 2: ■
  Centroids: 0,1,2,...
```

## 📁 Project Structure

```
kmeans-from-scratch/
├── README.md                           # This file
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── kmeans.py                      # Core K-means implementation
│   └── visualization.py               # ASCII plotting utilities
├── examples/
│   ├── simple_2d_clustering.py        # Interactive demo script
│   └── sample_data_generator.py       # Data generation utilities
└── tests/
    └── test_kmeans.py                 # Unit tests
```

## 🎓 How K-means Works

K-means clustering partitions data into k clusters by iteratively:

1. **Initialize**: Select k random points as initial cluster centers
2. **Assign**: Assign each point to its nearest cluster center
3. **Update**: Recalculate centers as mean of assigned points
4. **Repeat**: Continue until convergence

### Visual Algorithm Flow

```
     Start
        │
        ▼
   Initialize k
    centroids
        │
        ▼
   ┌────────┐
   │ Assign │ ←─────┐
   │ points │       │
   └────┬───┘       │
        │           │
        ▼           │
   ┌────────┐       │
   │ Update │       │
   │centers │       │
   └────┬───┘       │
        │           │
        ▼           │
    Converged? ─No──┘
        │
       Yes
        │
        ▼
      Done
```

## 🔧 API Reference

### KMeans Class

```python
KMeans(n_clusters=3, max_iters=100, tolerance=1e-4, random_state=None)
```

**Parameters:**
- `n_clusters` (int): Number of clusters to form
- `max_iters` (int): Maximum iterations before stopping
- `tolerance` (float): Convergence threshold for centroid movement
- `random_state` (int): Seed for reproducible results

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `fit(data)` | Fit model to data | self |
| `predict(data)` | Predict clusters for new data | labels |
| `fit_predict(data)` | Fit and return labels | labels |
| `inertia(data)` | Calculate sum of squared distances | float |

### Visualization Functions

```python
plot_clusters(data, labels, centroids, width=60, height=20)
```
Creates ASCII visualization of clustered data points.

```python
print_cluster_statistics(data, labels, centroids)
```
Displays detailed statistics for each cluster.

## 🧪 Running Tests

Execute the test suite to verify implementation:

```bash
python tests/test_kmeans.py
```

Expected output:
```
==================================================
RUNNING K-MEANS UNIT TESTS
==================================================

Testing K-means fit...
✓ K-means fit test passed
Testing K-means predict...
✓ K-means predict test passed
Testing K-means convergence...
✓ K-means convergence test passed (converged in 3 iterations)
Testing inertia calculation...
✓ Inertia test passed
Testing edge cases...
✓ Edge cases test passed

==================================================
ALL TESTS PASSED! ✓
==================================================
```

## 💡 Examples

### Example 1: Basic Clustering

```python
from src.kmeans import KMeans

# Simple 2D data
data = [
    [1, 2], [1.5, 1.8], [5, 8], 
    [8, 8], [1, 0.6], [9, 11]
]

# Cluster into 2 groups
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(data)

print(f"Cluster assignments: {labels}")
print(f"Centroids: {kmeans.centroids}")
```

### Example 2: Interactive Mode

Run the interactive example to experiment:

```bash
python examples/simple_2d_clustering.py
```

Then choose:
- Option 1: Generate new random data
- Option 2: Try different k values
- Option 3: Exit

### Example 3: Custom Data Generation

```python
from examples.sample_data_generator import generate_gaussian_cluster

# Create a single cluster
cluster = generate_gaussian_cluster(
    center=[5, 5],      # Center point
    spread=1.0,         # Standard deviation
    n_points=50         # Number of points
)
```

## 📈 Understanding Results

### Interpreting ASCII Plots

- **Symbols** (●, ▲, ■, ♦, ★): Different clusters
- **Numbers** (0, 1, 2...): Cluster centroids
- **Axes**: Shows data range for reference

### Quality Metrics

**Inertia**: Sum of squared distances from points to their centroids
- Lower is generally better
- Decreases with more clusters
- Use "elbow method" to find optimal k

**Convergence**: Algorithm stops when:
- Centroids move less than tolerance
- Maximum iterations reached

## 🎯 When to Use K-means

### ✅ Good Use Cases

- **Customer Segmentation**: Group customers by behavior
- **Image Compression**: Reduce color palette
- **Anomaly Detection**: Find outliers as small clusters
- **Document Clustering**: Group similar texts
- **Geographic Clustering**: Partition spatial data

### ⚠️ Limitations

| Limitation | Description | Alternative |
|------------|-------------|-------------|
| Fixed k | Must specify clusters in advance | DBSCAN, HDBSCAN |
| Spherical assumption | Assumes round clusters | Spectral clustering |
| Sensitive to scale | Features need normalization | Standardize data first |
| Local minima | Results depend on initialization | K-means++, multiple runs |
| Outlier sensitive | Outliers affect centroids | K-medoids |

## 🔬 Algorithm Complexity

- **Time Complexity**: O(n × k × i × d)
  - n = number of points
  - k = number of clusters  
  - i = iterations to converge
  - d = dimensions

- **Space Complexity**: O(n × d + k × d)

## 🚀 Extending the Code

### Ideas for Enhancement

1. **K-means++ Initialization**
```python
def initialize_plus_plus(self, data):
    """Smarter initialization for better results"""
    # Pick first center randomly
    # Pick next centers proportional to squared distance
```

2. **Silhouette Score**
```python
def silhouette_score(self, data, labels):
    """Measure clustering quality (-1 to 1)"""
    # Calculate intra-cluster and inter-cluster distances
```

3. **Elbow Method**
```python
def find_optimal_k(data, k_range):
    """Find optimal number of clusters"""
    # Plot inertia vs k, look for "elbow"
```

4. **Mini-batch K-means**
```python
def mini_batch_update(self, data, batch_size):
    """Update centroids using random batches"""
    # Faster for large datasets
```

## 📖 Educational Resources

### Concepts Demonstrated

- **Algorithm Design**: Clean, modular implementation
- **OOP Principles**: Proper class structure and encapsulation
- **Testing Practices**: Comprehensive unit tests
- **Documentation**: Clear docstrings and comments
- **Visualization**: Creative ASCII plotting without dependencies

### Learning Exercises

1. **Implement Different Metrics**: Try Manhattan or Cosine distance
2. **Add Preprocessing**: Implement data normalization
3. **Visualization Enhancement**: Add 3D ASCII projection
4. **Performance Analysis**: Time different initialization methods
5. **Parallel Processing**: Use `multiprocessing` for large datasets

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add more tests
- 🎨 Enhance visualizations

## 📚 References

### Academic Papers

1. MacQueen, J. B. (1967). "Some Methods for classification and Analysis of Multivariate Observations"
2. Lloyd, S. P. (1982). "Least squares quantization in PCM"
3. Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding"

### Books

- *Pattern Recognition and Machine Learning* by Christopher Bishop
- *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman
- *Introduction to Data Mining* by Tan, Steinbach, and Kumar

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```

## 🙏 Acknowledgments

Special thanks to the machine learning community for decades of research making clustering accessible to everyone.

---

<div align="center">

**Made with ❤️ for learners everywhere**

*"In the middle of difficulty lies opportunity."* - Albert Einstein

[⬆ Back to top](#k-means-clustering-from-scratch-)

</div>
