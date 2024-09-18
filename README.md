# **He-Li Graph Convolutional Neural Network: Deep and Enhanced Adaptation of Interest Modeling for Personalized Recommender System**

Graph Neural Network (GNN) has achieved magnificent success in learning representations of nodes in graphs. However, these methods lack the flexibility to thoroughly explore all potential meta-paths and efficiently identify the most relevant ones for a specific target, which negatively impacts both the effectiveness and interpretability of the models.

Moreover, many approaches require the generation of intermediate dense graphs based on meta-paths, further increasing computational overhead and hindering scalability. To address these challenges, our paper presents the He-Li Graph Convolutional Neural Network (He-LiGCN), a lightweight framework for heterogeneous graphs.
Specifically, we adopt the HeteroMILE for handling complex graph structures, which allows for more efficient multi-level representation learning.
For embedding, we utilize GATNE-T (Inductive General Attributed Multiplex Heterogeneous Network Embedding), a powerful method that captures rich structural and attribute information across different types of edges in the graph.
To mitigate the issue of over-smoothing in deep graph networks, we employ the APPNP (Approximate Personalized Propagation of Neural Predictions) technique. APPNP combines with Personalized PageRank propagation, allowing for controlled information diffusion while preserving the original features of nodes. This approach ensures that the model retains discriminative power in deeper layers by balancing between information from neighboring nodes and each node's original characteristics.

As a result, the model avoids losing important variations in the data, leading to better scalability and more accurate recommendations. These enhancements collectively enable the He-LiGCN to not only handle large-scale heterogeneous graphs efficiently but also deliver more personalized and interpretable recommendations.
