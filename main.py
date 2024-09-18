import argparse
import networkx as nx
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple

from config import load_config
from coarsen import coarsen_graph
from embed import GATNET
from refine_model import refine_embeddings
from eval_embed import evaluate_embeddings
from utils import load_graph, save_embeddings, save_results, load_labels

def parse_args():
    parser = argparse.ArgumentParser(description="HeteroMILE: Multi-Level Graph Representation Learning for Heterogeneous Graphs")
    
    # Data paths
    parser.add_argument('--graph_path', type=str, default='data/graph.txt', help='Path to the input graph file')
    parser.add_argument('--labels_path', type=str, default='data/labels.txt', help='Path to the node labels file')
    parser.add_argument('--features_path', type=str, default='data/features.txt', help='Path to the node features file')
    parser.add_argument('--output_path', type=str, default='results/', help='Path to save output files')

    # Graph coarsening parameters
    parser.add_argument('--num_coarse_levels', type=int, default=3, help='Number of coarsening levels')
    parser.add_argument('--coarsen_method', type=str, default='metis', choices=['metis', 'louvain', 'spectral'], help='Method for graph coarsening')

    # Embedding parameters
    parser.add_argument('--embed_method', type=str, default='deepwalk', choices=['deepwalk', 'node2vec', 'gatne-t'], help='Method for base embedding')
    parser.add_argument('--embed_dim', type=int, default=64, help='Dimension of the embeddings')
    parser.add_argument('--walk_length', type=int, default=80, help='Length of each random walk')
    parser.add_argument('--num_walks', type=int, default=10, help='Number of random walks per node')
    parser.add_argument('--window_size', type=int, default=10, help='Context window size for skip-gram')

    # Refinement parameters
    parser.add_argument('--num_refinement_layers', type=int, default=2, help='Number of refinement layers')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension in refinement model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for refinement')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for refinement')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for refinement')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

    # Evaluation parameters
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test data')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index, -1 for CPU')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config()

    # Update config with command-line arguments
    config.update(vars(args))

    # Load original heterogeneous graph
    G = load_graph(config.graph_path)

    # Load node labels
    labels = load_labels(config.labels_path)

    # Step 1: Graph Coarsening
    print("Performing graph coarsening...")
    coarsened_graphs = coarsen_graph(G, levels=config.num_coarse_levels)

    # Step 2: Base Embedding
    print("Generating base embeddings...")
    coarse_graph = coarsened_graphs[-1]
    node_types = set(nx.get_node_attributes(coarse_graph, 'type').values())
    edge_types = set((G.nodes[e[0]]['type'], G.nodes[e[1]]['type']) for e in G.edges())

    gatnet = GATNET(num_nodes=coarse_graph.number_of_nodes(),
                    num_edge_types=len(edge_types),
                    embedding_dim=config.embed_dim,
                    edge_embedding_dim=config.embed_dim // 2,
                    num_attention_heads=4)

    coarse_embeddings = {}
    for node_type in node_types:
        nodes = [n for n, d in coarse_graph.nodes(data=True) if d['type'] == node_type]
        neighbors = {et: [list(coarse_graph.neighbors(n)) for n in nodes] for et in edge_types if et[0] == node_type}
        coarse_embeddings[node_type] = gatnet.forward(tf.constant(nodes), neighbors)[node_type]

    # Step 3: Refinement
    print("Refining embeddings...")
    matching_matrices = []
    adjacencies = []
    node_features = []
    edge_features = []

    for i in range(len(coarsened_graphs) - 1, 0, -1):
        coarse_g = coarsened_graphs[i]
        fine_g = coarsened_graphs[i-1]
        
        # Create matching matrices
        matching = {(nt, nt): nx.adj_matrix(nx.bipartite.generic_weighted_projected_graph(
            nx.Graph([(u, v) for u, v in fine_g.edges() if fine_g.nodes[u]['type'] == nt and coarse_g.nodes[v]['type'] == nt]),
            nodes=[n for n in fine_g.nodes() if fine_g.nodes[n]['type'] == nt]
        )) for nt in node_types}
        matching_matrices.append(matching)
        
        # Create adjacency matrices
        adj = {(et[0], et[1]): nx.adj_matrix(nx.subgraph(fine_g, [n for n in fine_g.nodes() if fine_g.nodes[n]['type'] in et])) for et in edge_types}
        adjacencies.append(adj)
        
        # Create node and edge features (dummy features for this example)
        node_features.append({nt: np.random.rand(fine_g.number_of_nodes(), 10) for nt in node_types})
        edge_features.append({et: nx.adj_matrix(fine_g) for et in edge_types})

    refined_embeddings = refine_embeddings(coarse_embeddings, matching_matrices, adjacencies, node_features, edge_features)

    # Step 4: Evaluation
    print("Evaluating embeddings...")
    edge_list = {et: np.array([(u, v) for u, v in G.edges() if G.nodes[u]['type'] == et[0] and G.nodes[v]['type'] == et[1]]) for et in edge_types}
    results = evaluate_embeddings(refined_embeddings, labels, edge_list)

    # Save results and embeddings
    save_results(results, config.output_path)
    save_embeddings(refined_embeddings, f"{config.output_path}_embeddings.txt")

    print("Evaluation results:")
    for task, task_results in results.items():
        print(f"\n{task.upper()} RESULTS:")
        for key, value in task_results.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for metric, score in value.items():
                    print(f"    {metric}: {score:.4f}")
            else:
                print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()