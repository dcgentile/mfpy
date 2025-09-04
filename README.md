# MFPy: Time Series Clustering

A Python package for unsupervised time series clustering that conforms to scikit-learn standards.

## Features

- Scikit-learn compatible time series clustering algorithms
- Multiple distance metrics for time series (DTW, Euclidean, Correlation)
- Preprocessing utilities for time series data
- K-means and DBSCAN variants for time series clustering
- Synthetic time series generation for testing

## Installation

```bash
pip install mfpy
```

For development:

```bash
git clone https://github.com/yourusername/MFPy.git
cd MFPy
pip install -e .[dev]
```

## Quick Start

```python
import numpy as np
from mfpy import TimeSeriesKMeans, generate_synthetic_timeseries

# Generate synthetic time series data
X = generate_synthetic_timeseries(n_series=100, length=50, n_clusters=3)

# Fit clustering model
model = TimeSeriesKMeans(n_clusters=3, metric='dtw')
labels = model.fit_predict(X)

print(f"Cluster labels: {labels}")
```

## License

MIT License