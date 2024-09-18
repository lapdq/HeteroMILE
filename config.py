import os

class Config:
    def __init__(self):
        # Data paths
        self.graph_path = 'data/graph.txt'
        self.labels_path = 'data/labels.txt'
        self.features_path = 'data/features.txt'
        self.output_path = 'results/'

        # Graph coarsening parameters
        self.num_coarse_levels = 3
        self.coarsen_method = 'metis'  # Options: 'metis', 'louvain', 'spectral'

        # Embedding parameters
        self.embed_method = 'deepwalk'  # Options: 'deepwalk', 'node2vec', 'gatne-t'
        self.embed_dim = 128
        self.walk_length = 80
        self.num_walks = 10
        self.window_size = 10

        # Refinement parameters
        self.num_refinement_layers = 2
        self.hidden_dim = 64
        self.learning_rate = 0.01
        self.num_epochs = 200
        self.batch_size = 512
        self.dropout = 0.5

        # Evaluation parameters
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

    def create_dirs(self):
        """Create necessary directories."""
        os.makedirs(self.output_path, exist_ok=True)

    def update(self, params):
        """Update config with new parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")

def load_config(config_path: str = None) -> Config:
    """
    Load configuration from a JSON file if provided, else return default config.
    """
    config = Config()
    if config_path:
        import json
        with open(config_path, 'r') as f:
            params = json.load(f)
        config.update(params)
    return config