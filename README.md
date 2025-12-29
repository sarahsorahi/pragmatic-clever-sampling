# pragmatic-clever-sampling

## Understanding Why Augmented Data Improves Pragmatic Classification

This repository investigates why synthetic  data improve supervised classification of discourse–pragmatic functions, even when such data do not reproduce the pragmatic structure observed in real usage. The focus is on the English lexical item *look* and its pragmatic functions: **Interjection (INTJ)**, **Discourse Marker (DM)**, **Directive (DIR)**, and **AS**.

Rather than treating augmentation as uniformly beneficial, the project adopts a structure-aware approach that combines unsupervised clustering with controlled sampling strategies to isolate which types of synthetic examples contribute to performance gains.

---

## Project Structure

The project consists of two main stages:

1. **Unsupervised diagnostic analysis** 
2. **Controlled supervised experiments with different sampling strategies**

---

## Stage 1: Clustering-Based Diagnostic Analysis (Completed)

Unsupervised clustering was performed using transformer-based sentence embeddings and HDBSCAN to examine the natural structure of pragmatic functions. This analysis revealed that:

- Real data define tight pragmatic cores.
- Synthetic data often fail to reproduce these cores, especially for INTJ and DIR.
- Alignment between real and synthetic data is function-dependent:
  - DM shows strong overlap.
  - INTJ and DIR show little to no overlap.

This stage is fully documented in the companion repository:

👉 https://github.com/sarahsorahi/look-function-clustering

---

## Stage 2: Clever Sampling for Supervised Classification (Current)

Building on the clustering analysis, this repository evaluates four training conditions under a strictly controlled supervised setup. All conditions use:

- the same model and hyperparameters,
- a 50–50 split of real data into training and test sets,
- a real-only test set (held constant across conditions),
- no class-weighted loss or additional penalties,
- Macro F1 as the evaluation metric.

The only difference between conditions is how the training data are constructed.

---

## Training Conditions

### 1. Real-only 

Training uses only real (attested) data.  
This condition measures how well the pragmatic core alone supports classification under data sparsity.

---

### 2. Real + All Synthetic 

Training uses real data plus all available synthetic data for INTJ, DM, and DIR.  
AS remains real-only, as no synthetic data exist for this class.

This condition replicates earlier findings that augmentation improves Macro F1, but does not explain *why*.

---

### 3. Core-Anchored Sampling 

Training uses real data plus only those synthetic examples that fall inside clusters anchored by real data.

Empirically, this condition:
- collapses to real-only for INTJ and DIR,
- retains some synthetic data for DM.

It serves as a diagnostic step, showing that reproducing the pragmatic core is insufficient to explain augmentation gains.

---

### 4. Boundary Sampling 

Training uses real data plus boundary-sampled synthetic examples—synthetic data that lie near, but not inside, real-data clusters, excluding clear noise.

Boundary sampling:
- reintroduces synthetic data for INTJ and DIR in a principled way,
- preserves DM behavior,
- leaves AS unchanged (real-only).

This condition directly tests the hypothesis that augmentation helps via boundary-level information rather than core reproduction.

---

