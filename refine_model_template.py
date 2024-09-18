import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple

class HGCN:
    def __init__(self, node_types: List[str], input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        self.node_types = node_types
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> Dict[str, tf.Variable]:
        weights = {}
        for node_type in self.node_types:
            weights[f"W_self_{node_type}"] = tf.Variable(
                tf.random.normal([self.input_dim, self.hidden_dim]), name=f"W_self_{node_type}")
            for neighbor_type in self.node_types:
                weights[f"W_{neighbor_type}_to_{node_type}"] = tf.Variable(
                    tf.random.normal([self.input_dim, self.hidden_dim]), name=f"W_{neighbor_type}_to_{node_type}")
        
        weights["W_output"] = tf.Variable(tf.random.normal([self.hidden_dim, self.output_dim]), name="W_output")
        return weights

    def forward(self, embeddings: Dict[str, tf.Tensor], adjacency: Dict[Tuple[str, str], tf.Tensor]) -> Dict[str, tf.Tensor]:
        H = embeddings
        for _ in range(self.num_layers):
            H_next = {}
            for node_type in self.node_types:
                Z_self = tf.matmul(H[node_type], self.weights[f"W_self_{node_type}"])
                Z_neighbors = tf.zeros_like(Z_self)
                for neighbor_type in self.node_types:
                    if (neighbor_type, node_type) in adjacency:
                        Z_neighbor = tf.matmul(H[neighbor_type], self.weights[f"W_{neighbor_type}_to_{node_type}"])
                        Z_neighbors += tf.matmul(adjacency[(neighbor_type, node_type)], Z_neighbor)
                
                H_next[node_type] = tf.nn.relu(Z_self + Z_neighbors)
            H = H_next

        output = {node_type: tf.matmul(H[node_type], self.weights["W_output"]) for node_type in self.node_types}
        return output

def refine_embeddings(coarse_embeddings: Dict[str, np.ndarray], 
                      matching_matrix: Dict[Tuple[str, str], np.ndarray], 
                      adjacency: Dict[Tuple[str, str], np.ndarray],
                      num_epochs: int = 100) -> Dict[str, np.ndarray]:
    node_types = list(coarse_embeddings.keys())
    input_dim = next(iter(coarse_embeddings.values())).shape[1]
    hidden_dim = 64  # This can be adjusted
    output_dim = input_dim  # Output dimension same as input dimension

    # Convert numpy arrays to TensorFlow tensors
    coarse_embeddings_tf = {k: tf.constant(v, dtype=tf.float32) for k, v in coarse_embeddings.items()}
    matching_matrix_tf = {k: tf.constant(v, dtype=tf.float32) for k, v in matching_matrix.items()}
    adjacency_tf = {k: tf.constant(v, dtype=tf.float32) for k, v in adjacency.items()}

    # Project coarse embeddings to fine graph
    projected_embeddings = {
        node_type: tf.matmul(matching_matrix_tf[(node_type, node_type)], coarse_embeddings_tf[node_type])
        for node_type in node_types
    }

    model = HGCN(node_types, input_dim, hidden_dim, output_dim, num_layers=2)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            refined_embeddings = model.forward(projected_embeddings, adjacency_tf)
            loss = tf.reduce_sum([
                tf.reduce_mean(tf.square(refined_embeddings[node_type] - projected_embeddings[node_type]))
                for node_type in node_types
            ])

        gradients = tape.gradient(loss, model.weights.values())
        optimizer.apply_gradients(zip(gradients, model.weights.values()))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    return {k: v.numpy() for k, v in refined_embeddings.items()}

# def main():
#     # Example usage
#     coarse_embeddings = {
#         "user": np.random.rand(5, 16),
#         "item": np.random.rand(8, 16)
#     }
#     matching_matrix = {
#         ("user", "user"): np.random.rand(10, 5),
#         ("item", "item"): np.random.rand(15, 8)
#     }
#     adjacency = {
#         ("user", "item"): np.random.rand(10, 15),
#         ("item", "user"): np.random.rand(15, 10)
#     }

#     refined_embeddings = refine_embeddings(coarse_embeddings, matching_matrix, adjacency)
    
#     for node_type, embeddings in refined_embeddings.items():
#         print(f"Refined embeddings for {node_type}:")
#         print(embeddings)

# if __name__ == "__main__":
#     main()