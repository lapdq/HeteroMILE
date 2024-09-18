import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, Tuple, List
from scipy.sparse import csr_matrix

def node_classification(embeddings: Dict[str, np.ndarray], 
                        labels: Dict[str, np.ndarray],
                        train_ratio: float = 0.8) -> Dict[str, Dict[str, float]]:
    """
    Evaluate node classification performance for each node type.
    
    Args:
    embeddings: Dict of node embeddings for each node type
    labels: Dict of node labels for each node type
    train_ratio: Ratio of training data
    
    Returns:
    Dict of evaluation metrics for each node type
    """
    results = {}
    
    for node_type, emb in embeddings.items():
        if node_type in labels:
            X_train, X_test, y_train, y_test = train_test_split(emb, labels[node_type], 
                                                                train_size=train_ratio, 
                                                                stratify=labels[node_type])
            
            # Simple MLP classifier
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(emb.shape[1],)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(labels[node_type].shape[1], activation='softmax')
            ])
            
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
            macro_f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average='macro')
            
            results[node_type] = {
                'accuracy': accuracy,
                'macro_f1': macro_f1
            }
    
    return results

def link_prediction(embeddings: Dict[str, np.ndarray], 
                    edge_list: Dict[Tuple[str, str], np.ndarray],
                    test_ratio: float = 0.1) -> Dict[Tuple[str, str], float]:
    """
    Evaluate link prediction performance for each edge type.
    
    Args:
    embeddings: Dict of node embeddings for each node type
    edge_list: Dict of existing edges for each edge type
    test_ratio: Ratio of test edges
    
    Returns:
    Dict of AUC scores for each edge type
    """
    results = {}
    
    for edge_type, edges in edge_list.items():
        src_type, dst_type = edge_type
        src_emb = embeddings[src_type]
        dst_emb = embeddings[dst_type]
        
        # Split edges into train and test sets
        num_test = int(len(edges) * test_ratio)
        train_edges, test_edges = edges[:-num_test], edges[-num_test:]
        
        # Generate negative samples for test set
        negative_edges = np.column_stack((
            np.random.randint(0, src_emb.shape[0], num_test),
            np.random.randint(0, dst_emb.shape[0], num_test)
        ))
        
        # Combine positive and negative samples
        test_edges_all = np.vstack((test_edges, negative_edges))
        test_labels = np.hstack((np.ones(num_test), np.zeros(num_test)))
        
        # Calculate dot product similarity
        similarities = np.sum(src_emb[test_edges_all[:, 0]] * dst_emb[test_edges_all[:, 1]], axis=1)
        
        # Compute AUC score
        auc_score = roc_auc_score(test_labels, similarities)
        
        results[edge_type] = auc_score
    
    return results

def node_clustering(embeddings: Dict[str, np.ndarray], 
                    labels: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Evaluate node clustering quality using Normalized Mutual Information (NMI).
    
    Args:
    embeddings: Dict of node embeddings for each node type
    labels: Dict of true cluster labels for each node type
    
    Returns:
    Dict of NMI scores for each node type
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score
    
    results = {}
    
    for node_type, emb in embeddings.items():
        if node_type in labels:
            true_labels = np.argmax(labels[node_type], axis=1)
            n_clusters = len(np.unique(true_labels))
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            predicted_labels = kmeans.fit_predict(emb)
            
            nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
            
            results[node_type] = nmi_score
    
    return results

def evaluate_embeddings(embeddings: Dict[str, np.ndarray],
                        labels: Dict[str, np.ndarray],
                        edge_list: Dict[Tuple[str, str], np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of embeddings using multiple metrics.
    
    Args:
    embeddings: Dict of node embeddings for each node type
    labels: Dict of node labels for each node type
    edge_list: Dict of existing edges for each edge type
    
    Returns:
    Dict of evaluation results for different tasks
    """
    results = {}
    
    # Node Classification
    nc_results = node_classification(embeddings, labels)
    results['node_classification'] = nc_results
    
    # Link Prediction
    lp_results = link_prediction(embeddings, edge_list)
    results['link_prediction'] = lp_results
    
    # Node Clustering
    clustering_results = node_clustering(embeddings, labels)
    results['node_clustering'] = clustering_results
    
    return results

# def main():
#     # Example usage
#     node_types = ['user', 'item']
#     edge_types = [('user', 'item'), ('user', 'user'), ('item', 'item')]
    
#     # Generate dummy embeddings
#     embeddings = {nt: np.random.rand(100, 64) for nt in node_types}
    
#     # Generate dummy labels (one-hot encoded)
#     labels = {nt: np.eye(5)[np.random.choice(5, 100)] for nt in node_types}
    
#     # Generate dummy edge lists
#     edge_list = {et: np.column_stack((np.random.randint(0, 100, 500), np.random.randint(0, 100, 500))) 
#                  for et in edge_types}
    
#     # Evaluate embeddings
#     results = evaluate_embeddings(embeddings, labels, edge_list)
    
#     # Print results
#     for task, task_results in results.items():
#         print(f"\n{task.upper()} RESULTS:")
#         for key, value in task_results.items():
#             if isinstance(value, dict):
#                 print(f"  {key}:")
#                 for metric, score in value.items():
#                     print(f"    {metric}: {score:.4f}")
#             else:
#                 print(f"  {key}: {value:.4f}")

# if __name__ == "__main__":
#     main()