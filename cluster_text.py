from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

def clusterContext(split_transcript, n=10):
    # Preprocess the text with TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(split_transcript)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)

    # Calculate the distances from each cluster center to all points
    distances = cdist(kmeans.cluster_centers_, X.todense(), 'euclidean')

    # For each cluster, find the index of the point (text segment) with the minimum distance to the cluster center
    min_indices = distances.argmin(axis=1)
    closest_texts = [split_transcript[index] for index in min_indices]

    return closest_texts