# shbang
# Judah Tanninen
# Project 4: Clustering Pokemon

# Imports
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Helper functions



# Read the csv
df = pd.read_csv('Pokemon.csv')

# Get all possible pokemon types by getting all unique values for the "Type  1" column
types = df["Type 1"].unique()
# List of all the number columns, that we are going to use to fit.
number_cols=[]
# Choose a range of clusters to use, 2 - 15
cluster_range = list(range(2, 15))

# Dict of each type
stored_data = {}

# Loop over each type
for type in types:
    # Get the data for this specific type
    typeDf = df.query("Type 1 == " + type)
    # Limit the dataframe to just numerical columns
    typeDf = typeDf[number_cols]
    # Create a pipeline
    for n_clusters in cluster_range:
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('kmeans', KMeans(n_clusters=n_clusters))
        ])
        # Fit the data
        pipe.fit(typeDf)
