import networkx as nx
import numpy as np
import json
from typing import Dict, Tuple, List
import scipy
from scipy.sparse import csr_matrix
import tensorflow as tf

def load_graph(path: str) -> nx.Graph:
    """
    Load a graph from a file.
    The file should be in a format where each line represents an edge: node1 node2 [weight]
    """
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(parts[0], parts[1])
            elif len(parts) == 3:
                G.add_edge(parts[0], parts[1], weight=float(parts[2]))
    return G

def save_embeddings(embeddings: Dict[str, np.ndarray], path: str):
    """
    Save node embeddings to a file.
    """
    with open(path, 'w') as f:
        for node_type, emb in embeddings.items():
            for i, vec in enumerate(emb):
                f.write(f"{node_type}_{i} " + " ".join(map(str, vec)) + "\n")

def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    """
    Load node embeddings from a file.
    """
    embeddings = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            node_info = parts[0].split('_')
            node_type, node_id = '_'.join(node_info[:-1]), int(node_info[-1])
            vec = np.array(list(map(float, parts[1:])))
            if node_type not in embeddings:
                embeddings[node_type] = []
            embeddings[node_type].append((node_id, vec))
    
    for node_type in embeddings:
        embeddings[node_type] = np.array([x[1] for x in sorted(embeddings[node_type], key=lambda x: x[0])])
    
    return embeddings

def save_results(results: Dict, path: str):
    """
    Save evaluation results to a JSON file.
    """
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

def load_labels(path: str) -> Dict[str, np.ndarray]:
    """
    Load node labels from a file.
    The file should be in the format: node_type node_id label
    """
    labels = {}
    with open(path, 'r') as f:
        for line in f:
            node_type, node_id, label = line.strip().split()
            if node_type not in labels:
                labels[node_type] = []
            labels[node_type].append((int(node_id), int(label)))
    
    for node_type in labels:
        max_label = max(label for _, label in labels[node_type])
        label_array = np.zeros((len(labels[node_type]), max_label + 1))
        for i, (_, label) in enumerate(sorted(labels[node_type])):
            label_array[i, label] = 1
        labels[node_type] = label_array
    
    return labels

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not scipy.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """
    Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    """
    adj_normalized = normalize_adj(adj + scipy.sparse.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def create_minibatch(nodes: np.ndarray, batch_size: int):
    """
    Create minibatches from a list of nodes.
    """
    np.random.shuffle(nodes)
    batches = [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]
    return batches

def sparse_dropout(x, keep_prob, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)