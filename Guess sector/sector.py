# Guess sector based on stock performance

import pandas as pd
from sklearn.cluster import KMeans

# Load historical performance data from CSV file
data = pd.read_csv('./data/returns_test.csv')

# Drop any non-numeric columns (e.g., stock names)
data = data.select_dtypes(include='number')

# Preprocessing: fill NaN values with 0 or any other suitable method
data.fillna(0, inplace=True)

# Normalize the data to have zero mean and unit variance
normalized_data = (data - data.mean()) / data.std()

# Set the number of clusters (you can change this based on your preference)
num_clusters = 20

# Create K-Means model and fit it to the data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(normalized_data)

# Get the cluster labels for each stock
cluster_labels = kmeans.labels_

# Create a dictionary to store stocks in each cluster
stocks_by_cluster = {}
for i, label in enumerate(cluster_labels):
    if label not in stocks_by_cluster:
        stocks_by_cluster[label] = []
    stocks_by_cluster[label].append(data.index[i])

# Print the stocks in each cluster
for cluster, stocks in stocks_by_cluster.items():
    print(f"Cluster {cluster + 1}: {stocks}")
