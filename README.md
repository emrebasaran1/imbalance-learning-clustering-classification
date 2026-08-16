# Imbalance Learning: KNN Classification under Global and Cluster-Based Oversampling Strategies

This project studies how oversampling changes KNN classification on imbalanced binary datasets. Five KNN-family classifiers are evaluated on ten datasets using a no-oversampling baseline, SMOTE, G-SMOTE, and two cluster-based variants.

The experiment has two phases. Phase 1 applies SMOTE and G-SMOTE to the minority class as a global pool. Phase 2 first clusters the minority class, allocates synthetic samples with a custom four-factor rule, and then applies SMOTE or G-SMOTE inside each cluster. Clustering changes how the training set is constructed; final predictions are still produced by the same KNN classifiers.

## What is imbalance learning?

Imbalance learning concerns classification problems in which one class contains substantially more observations than another. The smaller class is called the minority class and is often the class of greater practical interest, such as a disease case, a fraudulent transaction, or a system failure.

A classifier optimized mainly for overall accuracy can perform well numerically while learning very little about the minority class. If 95% of the observations belong to one class, always predicting that class already produces 95% accuracy, despite completely failing to identify the minority class.

Distance-based classifiers such as KNN are also affected. When majority observations dominate the training set, they are more likely to dominate the neighborhood around a query point. Minority observations can therefore be outvoted even when they are close to that point.

Oversampling reduces this dominance by adding synthetic minority observations to the training set. It can improve minority recall and class balance, but it can also increase false positives. For this reason, this project evaluates Error, Geometric Mean, and F1 rather than relying on accuracy or error alone. Oversampling is applied only to training folds; test folds retain their original class distributions.

## Experimental design

### Phase 1: Global oversampling

- **No Oversampling:** original imbalanced training fold.
- **SMOTE:** synthetic samples generated from the minority class as one pool.
- **G-SMOTE:** synthetic samples generated inside geometric regions formed from the minority class as one pool.

### Phase 2: Cluster-based oversampling

Phase 2 adds **Cluster+SMOTE** and **Cluster+G-SMOTE**. Minority observations are partitioned with K-means, with the cluster count selected through an elbow-based procedure. Synthetic samples are distributed among clusters according to:

1. relative cluster size,
2. intra-cluster compactness,
3. distance from the majority class,
4. connectedness to other minority clusters.

Singleton clusters receive no synthetic samples. Sampling then takes place separately inside each eligible cluster.

## Methods

The five classifiers are Classical KNN, Modified KNN, two Fuzzy KNN variants (`m=2`), and Radius KNN. Their classification logic is implemented from scratch using Euclidean distance; scikit-learn is used only for Phase 2 K-means clustering.

Features are standardized independently inside each fold. Means and sample standard deviations are calculated from the training partition and then applied to both training and test data, preventing test-data leakage.

The project implementation of SMOTE interpolates between two randomly selected minority observations. G-SMOTE instead draws points inside a hypersphere whose radius is determined from a minority neighbor and the nearest observed majority point.

Each classifier uses dataset-specific `k` or radius values stored in `BEST_PARAMS`. These values remain fixed across all five training-set strategies. Performance is averaged across stratified 5-fold cross-validation, treating the minority class as positive.

## Results

Average results across the five classifiers and ten datasets:

| Training-set strategy | Error | G-Mean | F1 |
|---|---:|---:|---:|
| No Oversampling | **0.1326** | 0.7653 | 0.6770 |
| SMOTE | 0.1465 | **0.8214** | 0.6946 |
| G-SMOTE | 0.1409 | 0.8057 | 0.6966 |
| Cluster+SMOTE | 0.1443 | 0.8158 | 0.7016 |
| Cluster+G-SMOTE | **0.1375¹** | 0.8115 | **0.7046** |

¹ Lowest error among the oversampling strategies; No Oversampling has the lowest overall error.

- Oversampling improves G-Mean and F1 compared with the baseline.
- SMOTE achieves the highest average G-Mean.
- Cluster+G-SMOTE achieves the highest average F1 and the lowest error among oversampling methods.
- Cluster-based sampling is not universally superior; its benefit depends on dataset structure and the evaluation metric.

The main conclusion is therefore metric-dependent: global SMOTE is strongest for balanced recognition of both classes, while Cluster+G-SMOTE is strongest on average when minority-class F1 is prioritized.

## Datasets

Each workbook contains a `Description` sheet and the data sheet used by the program.

| Dataset | Source | Samples | Features | Class split (0 / 1) | IR |
|---|---|---:|---:|---:|---:|
| BCWD | [UCI 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 30 | 212 / 357 | 1.68x |
| CTO | [UCI 193](https://archive.ics.uci.edu/dataset/193/cardiotocography) | 2,126 | 21 | 134 / 1,992 | 14.87x |
| Ecoli | [UCI 39](https://archive.ics.uci.edu/dataset/39/ecoli) | 336 | 7 | 77 / 259 | 3.36x |
| Glass | [UCI 42](https://archive.ics.uci.edu/dataset/42/glass+identification) | 214 | 9 | 46 / 168 | 3.65x |
| Haberman | [UCI 43](https://archive.ics.uci.edu/dataset/43/haberman+s+survival) | 306 | 3 | 81 / 225 | 2.78x |
| Heart | [UCI 45](https://archive.ics.uci.edu/dataset/45/heart+disease) | 297 | 13 | 83 / 214 | 2.58x |
| PageBlock | [UCI 78](https://archive.ics.uci.edu/dataset/78/page+blocks+classification) | 5,473 | 10 | 560 / 4,913 | 8.77x |
| Parkinson | [UCI 174](https://archive.ics.uci.edu/dataset/174/parkinsons) | 195 | 22 | 48 / 147 | 3.06x |
| Pima | [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | 8 | 500 / 268 | 1.87x |
| Yeast | [UCI 110](https://archive.ics.uci.edu/dataset/110/yeast) | 1,484 | 8 | 163 / 1,321 | 8.10x |

## Repository structure

```text
.
├── data/              # Ten Excel datasets
├── docs/report.pdf    # Methodology and detailed results
├── src/main.py        # Experimental pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

## Running the project

```bash
git clone https://github.com/emrebasaran1/imbalance-learning-clustering-classification.git
cd imbalance-learning-clustering-classification
pip install -r requirements.txt
python src/main.py
```

The pure-Python distance calculations make the larger datasets noticeably slower. The script prints dataset progress and execution time; aggregate and dataset-level results are documented in the report.

## Report and license

See [`docs/report.pdf`](docs/report.pdf) for the complete methodology, formulas, parameter table, and dataset-level results. Code and documentation are released under the [MIT License](LICENSE); the included datasets remain subject to their original source terms.
