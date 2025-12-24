# pragmatic-clever-sampling

Clever Sampling for Pragmatic Data Augmentation

This repository investigates how different sampling strategies for augmented data affect supervised classification of discourse–pragmatic functions. Rather than treating all synthetic data as equally useful, the project explores whether **structure-aware sampling**, informed by unsupervised clustering, can improve learning while preserving the pragmatic structure defined by real data.

---

## Project Structure

This project consists of **two clearly separated parts**:

### Part 1: Unsupervised Clustering (completed)

The first part of the project performs **intra-function unsupervised clustering** of discourse–pragmatic uses of *look* using contextual embeddings, HDBSCAN, and UMAP.  
This analysis identifies:
- stable pragmatic cores based on attested data,
- peripheral and gradient uses,
- and noise points.

 **This part is completed and available here:**  
https://github.com/sarahsorahi/look-function-clustering

The outputs of that repository (cluster labels, noise annotations, and metadata) are treated as **fixed preprocessing artifacts** in the current project.

---

### Part 2: Structure-Aware Sampling for Supervised Learning (this repository)

This repository builds directly on the clustering results from Part 1 and focuses on **how synthetic data should be sampled during supervised training**.

The central assumption is:
- **Real (attested) data define the pragmatic core**, while
- **Augmented data mainly support generalization and stability**, especially under data sparsity.

The goal is to test whether selecting synthetic examples relative to the real-data core leads to better or more stable performance than using all synthetic data indiscriminately.

---

## Sampling Strategies

All sampling strategies:
- use the **same real training data**, and
- differ only in how synthetic examples are selected.

sampling/
├── base_sampling.py
├── core_anchored_sampling.py
├── boundary_sampling.py
├── cluster_balanced_sampling.py
└── noise_filtered_sampling.py


---

### `base_sampling.py`

Shared utilities for:
- separating real vs. synthetic data,
- enforcing synthetic-to-real ratios,
- ensuring consistency across experiments.

This file contains no experimental logic.

---

### `core_anchored_sampling.py` (primary strategy)

Keeps:
- all real data, and
- only synthetic examples that are structurally aligned with the real-data core  
  (e.g. non-noise points in clusters that contain real examples).

This strategy is directly motivated by the clustering results from Part 1.

---

### `boundary_sampling.py`

Selects synthetic examples near the **edges** of real-data clusters rather than the densest core, testing whether boundary cases help learning under data sparsity.

---

### `cluster_balanced_sampling.py`

Samples synthetic data evenly across clusters with per-cluster caps.  
This serves as a control strategy to distinguish structural alignment effects from simple balancing effects.

---

### `noise_filtered_sampling.py`

Removes synthetic examples labeled as noise by HDBSCAN, providing a minimal filtering baseline.

---

## Status

- Part 1 (clustering): **completed**
- Part 2 (sampling and supervised evaluation): **in progress**






