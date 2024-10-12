import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from sklearn.metrics.pairwise import haversine_distances


def geographical_distance(x=None, to_rad=True):
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights


def get_similarity_AQI(dist, thr=0.1, include_self=False, force_symmetric=False, sparse=False):
    theta = np.std(dist[:27, :27])  
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj


def get_adj_AQI_b():
    df = pd.read_csv("./data/air_quality/station_b.csv")
    df = df[['latitude', 'longitude']]
    res = geographical_distance(df, to_rad=False).values
    adj = get_similarity_AQI(res)
    return adj

def get_adj_AQI_t():
    df = pd.read_csv("./data/air_quality/station_t.csv")
    df = df[['latitude', 'longitude']]
    res = geographical_distance(df, to_rad=False).values
    adj = get_similarity_AQI(res)
    print(adj)
    return adj

def get_similarity_pems04():
    df = pd.read_csv('./data/pems/distance04.csv')
    # Find all unique nodes
    nodes = list(set(df['from']).union(set(df['to'])))
    nodes.sort()

    # Create an adjacency matrix filled with zeros
    adj_matrix = np.ones((len(nodes), len(nodes)))* np.inf

    # Populate the adjacency matrix
    for i, row in df.iterrows():
        from_node = nodes.index(row['from'])
        to_node = nodes.index(row['to'])
        adj_matrix[from_node, to_node] = row['cost']
    np.fill_diagonal(adj_matrix, 0)
    adj_matrix = adj_matrix[:170, :170]
    finite_dist = adj_matrix.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(adj_matrix / sigma))
    thr=0.1
    adj[adj < thr] = 0.
    return adj

def get_similarity_pems08():
    df = pd.read_csv('./data/pems/distance08.csv')
    # Find all unique nodes
    nodes = list(set(df['from']).union(set(df['to'])))
    nodes.sort()

    # Create an adjacency matrix filled with zeros
    adj_matrix = np.ones((len(nodes), len(nodes)))* np.inf

    # Populate the adjacency matrix
    for i, row in df.iterrows():
        from_node = nodes.index(row['from'])
        to_node = nodes.index(row['to'])
        adj_matrix[from_node, to_node] = row['cost']
    np.fill_diagonal(adj_matrix, 0)
    finite_dist = adj_matrix.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(adj_matrix / sigma))
    thr=0.1
    adj[adj < thr] = 0.
    return adj
        

