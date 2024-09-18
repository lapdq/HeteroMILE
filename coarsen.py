import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

class LSHMatcher:
	def __init__(self, k: int = 100):
		self.k = k
		self.hash_functions = [self._create_hash_function() for _ in range(k)]
	
	def _create_hash_function(self):
		return np.random.permutation(100000)  # Supposed maximum graph size
	
	def signature(self, neighbors: List[int]) -> Tuple[int, ...]:
		return tuple(min(self._hash_function[n] for n in neighbors) for self._hash_function in self.hash_functions)
	
	def match(self, graph: nx.Graph) -> Dict[str, List[List[int]]]:
		node_signatures = {}
		for node, data in graph.nodes(data=True):
			node_type = data.get('type', 'default')
			neighbors = list(graph.neighbors(node))
			node_signatures[node] = (node_type, self.signature(neighbors))
		
		matches = defaultdict(list)
		for node, (node_type, sig) in node_signatures.items():
			matches[node_type].append((node, sig))
		
		final_matches = defaultdict(list)
		for node_type, nodes in matches.items():
			nodes.sort(key=lambda x: x[1])
			current_match = []
			for node, sig in nodes:
				if not current_match or sig == current_match[-1][1]:
					current_match.append((node, sig))
				else:
					if len(current_match) > 1:
						final_matches[node_type].append([n for n, _ in current_match])
					current_match = [(node, sig)]
			if len(current_match) > 1:
				final_matches[node_type].append([n for n, _ in current_match])
		
		return final_matches

def coarsen_graph(graph: nx.Graph, levels: int = 3) -> List[nx.Graph]:
	coarsened_graphs = [graph]
	matcher = LSHMatcher()
	
	for _ in range(levels):
		current_graph = coarsened_graphs[-1]
		matches = matcher.match(current_graph)
		
		new_graph = nx.Graph()
		node_mapping = {}
		
		for node_type, match_groups in matches.items():
			for i, group in enumerate(match_groups):
				supernode = f"{node_type}_super_{i}"
				new_graph.add_node(supernode, type=node_type)
				for node in group:
					node_mapping[node] = supernode
		
		for u, v, data in current_graph.edges(data=True):
			super_u = node_mapping.get(u, u)
			super_v = node_mapping.get(v, v)
			if super_u != super_v:
				if new_graph.has_edge(super_u, super_v):
					new_graph[super_u][super_v]['weight'] += data.get('weight', 1)
				else:
					new_graph.add_edge(super_u, super_v, weight=data.get('weight', 1))
		
		for node in current_graph.nodes():
			if node not in node_mapping:
				new_graph.add_node(node, **current_graph.nodes[node])
				for neighbor in current_graph.neighbors(node):
					super_neighbor = node_mapping.get(neighbor, neighbor)
					new_graph.add_edge(node, super_neighbor, weight=current_graph[node][neighbor].get('weight', 1))
		
		coarsened_graphs.append(new_graph)
	
	return coarsened_graphs

# def main():
# 	# Example usage
# 	G = nx.karate_club_graph()
# 	nx.set_node_attributes(G, 'person', 'type')

# 	coarsened_graphs = coarsen_graph(G, levels=3)

# 	for i, graph in enumerate(coarsened_graphs):
# 		print(f"Level {i}: Nodes = {graph.number_of_nodes()}, Edges = {graph.number_of_edges()}")

# if __name__ == "__main__":
# 	main()