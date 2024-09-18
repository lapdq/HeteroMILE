import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import csr_matrix

class HGCN(tf.keras.Model):
    def __init__(self, node_types: List[str], input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, 
                 node_feature_dims: Dict[str, int], edge_feature_dims: Dict[Tuple[str, str], int]):
        super(HGCN, self).__init__()
        self.node_types = node_types
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.node_feature_dims = node_feature_dims
        self.edge_feature_dims = edge_feature_dims

        self.weights = self._initialize_weights()
        self.node_feature_projections = self._initialize_node_feature_projections()
        self.edge_feature_projections = self._initialize_edge_feature_projections()

    def _initialize_weights(self) -> Dict[str, tf.Variable]:
        weights = {}
        for layer in range(self.num_layers):
            for node_type in self.node_types:
                weights[f"W_self_{node_type}_{layer}"] = self.add_weight(
                    shape=(self.hidden_dim, self.hidden_dim),
                    initializer='glorot_uniform',
                    name=f"W_self_{node_type}_{layer}")
                for neighbor_type in self.node_types:
                    weights[f"W_{neighbor_type}_to_{node_type}_{layer}"] = self.add_weight(
                        shape=(self.hidden_dim, self.hidden_dim),
                        initializer='glorot_uniform',
                        name=f"W_{neighbor_type}_to_{node_type}_{layer}")
        
        weights["W_output"] = self.add_weight(
            shape=(self.hidden_dim, self.output_dim),
            initializer='glorot_uniform',
            name="W_output")
        return weights

    def _initialize_node_feature_projections(self) -> Dict[str, tf.keras.layers.Dense]:
        return {node_type: tf.keras.layers.Dense(self.hidden_dim) 
                for node_type, dim in self.node_feature_dims.items()}

    def _initialize_edge_feature_projections(self) -> Dict[Tuple[str, str], tf.keras.layers.Dense]:
        return {edge_type: tf.keras.layers.Dense(self.hidden_dim) 
                for edge_type, dim in self.edge_feature_dims.items()}

    def call(self, inputs: Dict[str, tf.Tensor], 
             adjacency: Dict[Tuple[str, str], tf.SparseTensor], 
             node_features: Dict[str, tf.Tensor],
             edge_features: Dict[Tuple[str, str], tf.SparseTensor]) -> Dict[str, tf.Tensor]:
        H = {node_type: self.node_feature_projections[node_type](node_features[node_type]) 
             for node_type in self.node_types}
        
        for layer in range(self.num_layers):
            H_next = {}
            for node_type in self.node_types:
                Z_self = tf.matmul(H[node_type], self.weights[f"W_self_{node_type}_{layer}"])
                Z_neighbors = tf.zeros_like(Z_self)
                for neighbor_type in self.node_types:
                    if (neighbor_type, node_type) in adjacency:
                        Z_neighbor = tf.matmul(H[neighbor_type], self.weights[f"W_{neighbor_type}_to_{node_type}_{layer}"])
                        edge_projection = self.edge_feature_projections[(neighbor_type, node_type)](edge_features[(neighbor_type, node_type)])
                        Z_neighbors += tf.sparse.sparse_dense_matmul(adjacency[(neighbor_type, node_type)], Z_neighbor * edge_projection)
                
                H_next[node_type] = tf.nn.relu(Z_self + Z_neighbors)
            H = H_next

        output = {node_type: tf.matmul(H[node_type], self.weights["W_output"]) for node_type in self.node_types}
        return output

def refine_embeddings(coarse_embeddings: Dict[str, np.ndarray], 
                      matching_matrices: List[Dict[Tuple[str, str], csr_matrix]], 
                      adjacencies: List[Dict[Tuple[str, str], csr_matrix]],
                      node_features: List[Dict[str, np.ndarray]],
                      edge_features: List[Dict[Tuple[str, str], csr_matrix]],
                      num_epochs: int = 100,
                      batch_size: int = 1024) -> Dict[str, np.ndarray]:
    node_types = list(coarse_embeddings.keys())
    input_dim = next(iter(coarse_embeddings.values())).shape[1]
    hidden_dim = 64  # This can be adjusted
    output_dim = input_dim  # Output dimension same as input dimension

    node_feature_dims = {node_type: features.shape[1] for node_type, features in node_features[-1].items()}
    edge_feature_dims = {edge_type: features.shape[1] for edge_type, features in edge_features[-1].items()}

    model = HGCN(node_types, input_dim, hidden_dim, output_dim, num_layers=len(matching_matrices),
                 node_feature_dims=node_feature_dims, edge_feature_dims=edge_feature_dims)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # Convert sparse matrices to SparseTensor
    adjacencies_tf = [{k: tf.sparse.from_sparse_tensor_value(tf.sparse.SparseTensor(
        indices=np.array([v.row, v.col]).T,
        values=v.data,
        dense_shape=v.shape
    )) for k, v in adj.items()} for adj in adjacencies]

    edge_features_tf = [{k: tf.sparse.from_sparse_tensor_value(tf.sparse.SparseTensor(
        indices=np.array([v.row, v.col]).T,
        values=v.data,
        dense_shape=v.shape
    )) for k, v in ef.items()} for ef in edge_features]

    # Refinement from coarsest to finest
    current_embeddings = {k: tf.constant(v, dtype=tf.float32) for k, v in coarse_embeddings.items()}
    for level in range(len(matching_matrices)):
        print(f"Refining level {level}")
        matching_matrix_tf = {k: tf.sparse.from_sparse_tensor_value(tf.sparse.SparseTensor(
            indices=np.array([v.row, v.col]).T,
            values=v.data,
            dense_shape=v.shape
        )) for k, v in matching_matrices[level].items()}

        # Project embeddings to finer level
        projected_embeddings = {
            node_type: tf.sparse.sparse_dense_matmul(matching_matrix_tf[(node_type, node_type)], current_embeddings[node_type])
            for node_type in node_types
        }

        node_features_tf = {k: tf.constant(v, dtype=tf.float32) for k, v in node_features[level].items()}

        for epoch in range(num_epochs):
            total_loss = 0
            for node_type in node_types:
                num_nodes = projected_embeddings[node_type].shape[0]
                for i in range(0, num_nodes, batch_size):
                    batch_indices = tf.range(i, min(i + batch_size, num_nodes))
                    with tf.GradientTape() as tape:
                        refined_embeddings = model({
                            nt: tf.gather(projected_embeddings[nt], batch_indices) if nt == node_type else projected_embeddings[nt]
                            for nt in node_types
                        }, adjacencies_tf[level], node_features_tf, edge_features_tf[level])
                        loss = tf.reduce_mean(tf.square(refined_embeddings[node_type] - tf.gather(projected_embeddings[node_type], batch_indices)))
                        total_loss += loss

                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.numpy()}")

        # Update current embeddings for next level
        current_embeddings = model(projected_embeddings, adjacencies_tf[level], node_features_tf, edge_features_tf[level])

    return {k: v.numpy() for k, v in current_embeddings.items()}

def evaluate_embeddings(embeddings: Dict[str, np.ndarray], 
                        labels: Dict[str, np.ndarray], 
                        edges: Dict[Tuple[str, str], np.ndarray]) -> Dict[str, float]:
    results = {}

    # Node Classification
    for node_type, emb in embeddings.items():
        if node_type in labels:
            X_train, X_test, y_train, y_test = train_test_split(emb, labels[node_type], test_size=0.2, random_state=42)
            clf = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(labels[node_type].shape[1], activation='softmax')
            ])
            clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            clf.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            y_pred = clf.predict(X_test)
            results[f'node_classification_{node_type}'] = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    # Link Prediction
    for edge_type, edge_list in edges.items():
        src_type, dst_type = edge_type
        src_emb = embeddings[src_type]
        dst_emb = embeddings[dst_type]
        
        # Create negative samples
        num_edges = edge_list.shape[0]
        negative_edges = np.column_stack((
            np.random.randint(0, src_emb.shape[0], num_edges),
            np.random.randint(0, dst_emb.shape[0], num_edges)
        ))
        
        # Combine positive and negative samples
        all_edges = np.vstack((edge_list, negative_edges))
        labels = np.hstack((np.ones(num_edges), np.zeros(num_edges)))
        
        # Calculate dot product similarity
        similarities = np.sum(src_emb[all_edges[:, 0]] * dst_emb[all_edges[:, 1]], axis=1)
        
        results[f'link_prediction_{edge_type}'] = roc_auc_score(labels, similarities)

    return results


# def main():
#     # Example usage
#     node_types = ['user', 'item']
#     num_levels = 3
    
#     # Generate dummy data
#     coarse_embeddings = {nt: np.random.rand(10, 16) for nt in node_types}
#     matching_matrices = [{(nt, nt): csr_matrix(np.random.rand(20, 10)) for nt in node_types} for _ in range(num_levels)]
#     adjacencies = [{(nt1, nt2): csr_matrix(np.random.rand(20, 20)) for nt1 in node_types for nt2 in node_types} for _ in range(num_levels)]
#     node_features = [{nt: np.random.rand(20, 5) for nt in node_types} for _ in range(num_levels)]
#     edge_features = [{(nt1, nt2): csr_matrix(np.random.rand(20, 20, 3)) for nt1 in node_types for nt2 in node_types} for _ in range(num_levels)]
    
#     refined_embeddings = refine_embeddings(coarse_embeddings, matching_matrices, adjacencies, node_features, edge_features)
    
#     # Generate dummy labels and edges for evaluation
#     labels = {nt: np.eye(5)[np.random.choice(5, 20)] for nt in node_types}
#     edges = {(nt1, nt2): np.column_stack((np.random.randint(0, 20, 100), np.random.randint(0, 20, 100))) 
#              for nt1 in node_types for nt2 in node_types}
    
#     evaluation_results = evaluate_embeddings(refined_embeddings, labels, edges)
#     print("Evaluation Results:", evaluation_results)

# if __name__ == "__main__":
#     main()