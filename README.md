# pragmatic-clever-sampling

Clever Sampling for Pragmatic Data Augmentation

This repository investigates how different sampling strategies for augmented data affect supervised classification of discourse–pragmatic functions. Rather than treating all synthetic data as equally useful, the project explores whether **structure-aware sampling**, informed by unsupervised clustering, can improve learning while preserving the pragmatic structure defined by real data.

---

## Sampling Strategies

All sampling strategies keep the **same real training data** and differ only in how synthetic examples are selected.

---

### `core_anchored_sampling.py` (main strategy)
Keeps:
- all real data, and
- only synthetic examples that are structurally aligned with the real-data core  
  (e.g. non-noise points in clusters containing real examples).

This strategy is directly motivated by the clustering results and serves as the primary experimental condition.

---

### `boundary_sampling.py`
Selects synthetic examples near the **edges** of real-data clusters rather than the densest core, testing whether boundary cases help learning under sparsity.

---

### `cluster_balanced_sampling.py`
Samples synthetic data evenly across clusters with per-cluster caps.  
This serves as a control strategy to distinguish structural alignment effects from simple balancing effects.

---

### `noise_filtered_sampling.py`
Removes synthetic examples labeled as noise by HDBSCAN, providing a minimal filtering baseline.


