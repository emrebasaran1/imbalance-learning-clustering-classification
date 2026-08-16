# Imbalance Learning: KNN Classification with Clustering-Guided Oversampling

A from-scratch (no ML-classifier libraries) study of imbalanced binary classification: five KNN-family classifiers are evaluated on ten imbalanced datasets, comparing standard oversampling (SMOTE, G-SMOTE) against a clustering-guided oversampling strategy applied before SMOTE/G-SMOTE generation.

## What is imbalance learning?

Imbalance learning (or imbalanced learning) is the part of machine learning concerned with classification problems where one class has far more examples than the other, for instance fraud detection, disease diagnosis, or fault detection, where the case that matters most is also the rarest one. Standard classifiers are trained to maximize overall accuracy, so on an imbalanced dataset they can reach a high accuracy simply by predicting the majority class almost every time, while barely learning to recognize the minority class at all. This project deals with that problem in two ways: by oversampling, generating synthetic minority-class examples so the classifier sees a more balanced training set, and by evaluating with metrics (Geometric Mean, F1 score) that stay meaningful even when the classes are imbalanced, instead of relying on accuracy or error rate alone.

## Overview

Standard classifiers tend to favor the majority class when class sizes are unequal, and distance-based classifiers such as KNN are particularly sensitive to this because majority points can dominate a query point's neighborhood. This project studies how much oversampling helps, and whether adding a clustering step before oversampling helps further, across ten imbalanced datasets.

The study runs in two phases:

- **Phase 1** builds the full evaluation pipeline (standardization, stratified 5-fold cross-validation, per-dataset hyperparameter selection) and compares three training strategies: No Oversampling, SMOTE, and G-SMOTE.
- **Phase 2** adds a clustering step before oversampling: the minority class is split into clusters with K-means (cluster count chosen via the elbow method), synthetic points are distributed across clusters using a four-factor weighting rule, and SMOTE or G-SMOTE is then applied within each cluster. This adds two more strategies: Cluster+SMOTE and Cluster+G-SMOTE.

All five strategies are evaluated with the same classifiers, folds, and fixed hyperparameters, so any difference in results can be attributed to how the training set was constructed.

## Method summary

**Classifiers:** Classical KNN, Modified (distance-weighted) KNN, Fuzzy KNN (hard membership, m=2), Fuzzy KNN (distance-to-class-mean membership, m=2), and Radius-based KNN. All five are implemented from scratch using only `math.dist` for distance computation.

**SMOTE:** For each synthetic point, two minority points are drawn at random and a new point is placed at a random location on the line segment between them.

**G-SMOTE:** Generalizes SMOTE from a line segment to a full hypersphere. For a chosen minority center point, the sphere radius is capped at whichever is closer: the nearest same-class neighbor or the nearest majority point (so generated points cannot cross into the majority region). A point is then drawn uniformly inside that sphere. No truncation or deformation is applied, matching the configuration reported to work best in the source literature.

**Clustering-guided oversampling (Phase 2):** The minority class in each training fold is clustered with K-means; the number of clusters is chosen with the elbow method (SSE vs. cluster count), capped by the square root of the number of points that need to be generated. Each cluster then receives a share of the synthetic points proportional to a weight combining four factors: relative cluster size, intra-cluster tightness, distance from the majority class, and connectedness to the rest of the minority class. Singleton clusters are treated as outliers and receive no synthetic points. SMOTE or G-SMOTE is then run independently inside each cluster's point pool.

**Evaluation:** Each dataset is split into 5 stratified folds; oversampling is applied only to the training folds, never to the test fold. Metrics are Error rate, Geometric Mean (`sqrt(Sensitivity * Specificity)`), and F1 score, computed with the minority class as positive and averaged across folds. The per-dataset neighbor count `k` (or radius `r` for Radius KNN) is selected once per classifier via internal validation and then reused unchanged across all five training strategies.

## Datasets

All ten datasets are binary-classification versions of well-known UCI (and one Kaggle) datasets. Each `data/*.xlsx` file has a `Description` sheet with source and column details, and a data sheet with the standardized feature/label columns used by the code.

| Dataset | Source | Samples | Features | Class split (0 / 1) | Imbalance ratio |
|---|---|---|---|---|---|
| BCWD | UCI ML Repository, ID 17 | 569 | 30 | 212 / 357 | 1.68x |
| CTO | UCI ML Repository, ID 193 | 2126 | 21 | 134 / 1992 | 14.87x |
| Ecoli | UCI ML Repository, ID 39 | 336 | 7 | 77 / 259 | 3.36x |
| Glass | UCI ML Repository, ID 42 | 214 | 9 | 46 / 168 | 3.65x |
| Haberman | UCI ML Repository, ID 43 | 306 | 3 | 81 / 225 | 2.78x |
| Heart | UCI ML Repository, ID 45 | 297 | 13 | 83 / 214 | 2.58x |
| PageBlock | UCI ML Repository, ID 78 | 5473 | 10 | 560 / 4913 | 8.77x |
| Parkinson | UCI ML Repository, ID 174 | 195 | 22 | 48 / 147 | 3.06x |
| Pima | Kaggle / UCI, Pima Indians Diabetes | 768 | 8 | 500 / 268 | 1.87x |
| Yeast | UCI ML Repository, ID 110 | 1484 | 8 | 163 / 1321 | 8.1x |

## Report

[`docs/report.pdf`](docs/report.pdf) is the full project report: it explains the methodology in detail and documents the actual results, including per-dataset breakdowns for both phases (Appendix A and B) that are only summarized in this README.

## Project structure

```
.
├── data/            # 10 datasets, each an .xlsx with a Description sheet + data sheet
├── docs/
│   └── report.pdf    # full project report: methodology, formulas, and detailed results
├── src/
│   └── main.py       # full pipeline: data loading, classifiers, SMOTE/G-SMOTE, clustering, Phase 1 & 2 runners
├── requirements.txt
└── LICENSE
```

## Running it

```bash
pip install -r requirements.txt
python src/main.py
```

`src/main.py` loads all ten datasets from `data/`, then runs Phase 1 and Phase 2 sequentially, printing per-dataset progress and timing as it goes. The classifiers are implemented in pure Python without vectorization, so larger datasets (PageBlock, CTO, Yeast) take noticeably longer than the smaller ones; expect the full run to take a while on a single machine. K-means and elbow-method cluster-count selection (Phase 2 only) use `scikit-learn` and `kneed`; everything else has no ML-library dependency.

## License

MIT. See [LICENSE](LICENSE).
