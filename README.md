# MFPy: Time Series Clustering

A Python package for unsupervised change point detection and segment-based clustering

## Features

- Scikit-learn compatible time series clustering algorithms

## Installation
For development, initiate a virtual environment, clone, and install, like so:

```bash
python -m venv .venv
git clone https://github.com/dcgentile/mfpy.git
cd mfpy
pip install -e .[dev]
```

## Quick Start

```python
from mfpy.utils import label_series
from mfpy.clustering import SegmentBasedCluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import deeptime

filename = "../../../data/Langevin/langevin.txt"
data = np.loadtxt(filename)
data = np.reshape(data, (-1, 1)) # because this is 1D data we want to pass it as a column vector
sbc = SegmentBasedCluster() # optionally you can pass arguments to set hyperparameters
sbc.fit(data)
segment_labels = sbc.segment_labels_
point_labels = sbc.point_labels_
changes = sbc.change_points_
```

## License

MIT License
