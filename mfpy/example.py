#!/usr/bin/env python3

from mfpy.utils import label_series
from mfpy.clustering import SegmentBasedCluster
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

filename = "../data/Langevin/langevin.txt"
# filename = "../data/prinz/prinz.txt"
data = np.loadtxt(filename)
# T = data.shape[0]
T = 30000
x = data[:T]
w = 2**8
q = 0.75
# sbc = SegmentBasedCluster(w, q, method="spectral", num_clusters=3)
sbc = SegmentBasedCluster(w, q)
sbc.fit(x)

segment_labels = sbc.labels_
change_points = sbc.change_points_
labels = label_series(x, change_points, segment_labels)

df = pd.DataFrame({"time": range(T), "value": x, "label": labels})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="time", y="value", hue="label", palette="bright")
plt.show()

n_clusters = len(df["label"].unique())

g = sns.FacetGrid(df, col="label", col_wrap=3, height=4)
g.map(sns.histplot, "value", kde=True, stat="probability")
g.set_axis_labels("Value", "Frequency")
g.set_titles("Cluster {col_name}")
plt.show()
