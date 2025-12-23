# pragmatic-clever-sampling

Clever Sampling for Pragmatic Data Augmentation
This repository investigates how different sampling strategies for augmented data affect supervised classification of discourse–pragmatic functions. Rather than treating all synthetic data as equally useful, the project explores whether structure-aware sampling, informed by unsupervised clustering, can improve learning while preserving the pragmatic structure defined by real data.
The work builds on two observations:
(1) attested data form a stable core for each pragmatic function in embedding space, and
(2) augmented data consistently improve classification performance, despite being more dispersed and less structurally coherent.
Core Idea
Real data appear to define the pragmatic core of each function, while augmented data mainly populate the periphery. This repository tests the idea that augmentation works best when synthetic examples are selected relative to the real-data core, rather than used indiscriminately.
The goal is not to replace real data, but to use augmentation as a support mechanism that improves generalization and prevents class collapse under data sparsity.
Data and Setup
Task: classification of discourse–pragmatic functions of look (e.g. INTJ, DM, DIR)
Data:
Real (attested) examples
Synthetic (augmented) examples
Embeddings: transformer-based contextual embeddings
Clustering: HDBSCAN (used to identify cores, periphery, and noise)
Evaluation:
2-fold split on real data
5 random seeds
Macro F1 as the main metric
Class-weighted cross-entropy loss to address class imbalance
Sampling Strategies
All sampling strategies keep the same real training data and differ only in how synthetic examples are selected.
sampling/
├── base_sampling.py
├── core_anchored_sampling.py
├── boundary_sampling.py
├── cluster_balanced_sampling.py
└── noise_filtered_sampling.py
base_sampling.py
Shared utilities for:
separating real vs. synthetic data,
enforcing synthetic-to-real ratios,
ensuring consistency across experiments.
This file contains no experimental logic.
core_anchored_sampling.py (main strategy)
Keeps:
all real data, and
only synthetic examples that are structurally aligned with the real-data core (e.g. non-noise points in clusters containing real examples).
This strategy is directly motivated by the clustering results and serves as the primary experimental condition.
boundary_sampling.py
Selects synthetic examples near the edges of real-data clusters rather than the densest core.
This strategy explores whether boundary cases help the model learn decision regions under sparsity.
cluster_balanced_sampling.py
Samples synthetic data evenly across clusters, with per-cluster caps.
This acts as a control strategy to distinguish structural alignment effects from simple balancing effects.
