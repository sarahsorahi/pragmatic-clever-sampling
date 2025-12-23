# pragmatic-clever-sampling

Clever Sampling for Pragmatic Data Augmentation

This repository investigates how different sampling strategies for augmented data affect supervised classification of discourse–pragmatic functions. Rather than treating all synthetic data as equally useful, the project explores whether **structure-aware sampling**, informed by unsupervised clustering, can improve learning while preserving the pragmatic structure defined by real data.

---

## Motivation

Two consistent observations motivate this work:

1. Attested (real) data form a **stable pragmatic core** for each function in embedding space.
2. Augmented (synthetic) data consistently improve classification performance, despite being more dispersed and less structurally coherent.

These findings suggest that real data anchor pragmatic meaning, while augmented data mainly support learning by increasing coverage and stability. This repository tests whether augmentation works best when synthetic examples are selected **relative to the real-data core**, rather than used indiscriminately.

---

## Task and Data

- **Task:** Classification of discourse–pragmatic functions of *look*  
  (e.g. INTJ, DM, DIR)
- **Data:**
  - Real (attested) examples
  - Synthetic (augmented) examples
- **Embeddings:** Transformer-based contextual embeddings
- **Clustering:** HDBSCAN (used to identify cores, periphery, and noise)

---

## Experimental Setup

- Evaluation on **real data only**
- **2-fold split** on real data
- **5 random seeds**
- **Macro F1** as the main evaluation metric
- **Class-weighted cross-entropy loss** to address class imbalance and prevent minority-class collapse

---

## Sampling Strategies

All sampling strategies keep the **same real training data** and differ only in how synthetic examples are selected.

