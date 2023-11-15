# shbang
# Judah Tanninen
# Project 4: Clustering Pokemon

# Imports
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Ignoring memory leak warning on windows for kmeans
import warnings
warnings.filterwarnings('ignore')


# Read the csv
df = pd.read_csv('Pokemon.csv')

# Get all possible pokemon types by getting all unique values for the "Type  1" column
types = df["Type 1"].unique()
# List of all the number columns, that we are going to use to fit.
number_cols=["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
# Choose a range of clusters to use, 2 - 15
cluster_range = list(range(2, 15))

# Dict of each type
stored_data = {}

# Loop over each type
for type in types:
    print(type)
    print("-----------")
    # Get the data for this specific type
    typeDf = df.query("`Type 1` == '" + type + "'")
    # Limit the dataframe to just numerical columns
    typeDf = typeDf[number_cols]
    n_samples = len(typeDf.index) # Number of rows for this type, need this to cap the number of clusters we can use

    # Loop for each number of clusters.
    for n_clusters in cluster_range:
        if (n_clusters >= n_samples): continue # Skip if not enough data. (need more samples then clusters)
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('kmeans', KMeans(n_clusters=n_clusters, n_init="auto"))
        ])
        # Fit the data
        pipe.fit(typeDf)
        # Get just the kmeans part of the pipe, so we can get the labels from it.
        kmean = pipe[1]
        labels = kmean.labels_

        # Calculate the silhouette score for the data, this way we can see how good the number of clusters was
        score = silhouette_score(typeDf, labels)
        if (type not in stored_data or stored_data[type]["score"] < score): # Check if this score was better than previous, if so, update the stored data with it
            # Also, if we have no data, store it.
            stored_data[type] = {
                "labels": labels,
                "score": score,
                "n_clusters": n_clusters
            }
        # Print the score for this number of clusters
        print(str(n_clusters) + " clusters: " + str(score))
    # Print the best score and number of clusters
    print("best number of clusters: " + str(stored_data[type]["n_clusters"]))
    print("best score: " + str(stored_data[type]["score"]))
    print("")
# End loop

# Now, we have the results of all the kmeans and scores.

# Create a new array for the columns we want to print out.
printCols = ["Name"] + number_cols
# Loop through the types again
for type in types:
    print(type)
    print("-----------")
    typeDf = df.query("`Type 1` == '" + type + "'")
    # Get the saved data for this type
    typeData = stored_data[type]
    # Loop from 0 to the number of clusters
    for n_cluster in range(0, typeData["n_clusters"]):
        # Filter data for the current cluster
        cluster_data = typeDf[typeData["labels"] == n_cluster]
        # Get the averages for all the stats in this cluster.
        averages = np.mean(cluster_data[number_cols], axis=0)

        # All calculation now done, begin printing the last things.

        print("Cluster " + str(n_cluster))
        print(cluster_data[printCols])

        # Loop through the number columns and print all those averages
        for i, col in enumerate(number_cols):
            print("Mean " + col + ": " + str(averages[i]))
        print()