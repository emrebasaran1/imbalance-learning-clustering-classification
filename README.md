# Imbalance Learning: KNN Classification under Global and Cluster-Based Oversampling Strategies

This project investigates how different training-set construction strategies affect KNN-family classifiers on imbalanced binary classification problems. It compares a no-oversampling baseline with SMOTE and G-SMOTE, then examines how their behavior changes when the minority class is first divided into clusters and synthetic samples are generated within those local regions according to a custom allocation rule.

The study uses ten imbalanced datasets, five KNN-family classifiers, stratified 5-fold cross-validation, and three imbalance-sensitive evaluation metrics. The objective is not to identify a universally superior method, but to understand how global and cluster-based oversampling change the trade-off between overall error, class balance, and minority-class prediction quality.

## Research questions

1. How do SMOTE and G-SMOTE change the performance of KNN classifiers compared with training on the original imbalanced data?
2. How do the results change when the minority class is clustered first and synthetic samples are allocated and generated within those clusters?

Clustering does not replace classification in the second phase. The final predictions are still produced by the same five KNN-family classifiers. Clustering changes how the oversampled training set is constructed.

## What is imbalance learning?

Imbalance learning concerns classification problems in which one class contains substantially more observations than another. The smaller class is called the minority class and is often the class of greater practical interest, such as a disease case, a fraudulent transaction, or a system failure.

A classifier optimized mainly for overall accuracy can perform well numerically while learning very little about the minority class. For example, when 95% of the observations belong to one class, predicting the majority class for every observation already produces 95% accuracy. The resulting model appears successful according to accuracy, but it completely fails to identify the minority class.

Distance-based classifiers such as KNN are also affected by this problem. A prediction is determined by the observations surrounding a query point. When majority observations dominate the training set, they are more likely to dominate local neighborhoods as well. Minority observations can therefore be outvoted even when they are close to the query point.

Oversampling addresses this issue by adding synthetic minority observations to the training set. The purpose is not simply to increase the amount of data, but to reduce majority-class dominance and give the classifier a better opportunity to learn minority regions. Oversampling is applied only to the training folds in this project; the test folds retain their original class distributions.

This intervention creates a trade-off. It can improve minority recall and class balance while also increasing the number of majority observations incorrectly predicted as minority. For this reason, error rate alone is not sufficient. The project also evaluates Geometric Mean and F1 score, which reveal different aspects of minority-class performance.

## Experimental design

### Phase 1: Global oversampling

Phase 1 establishes the common evaluation pipeline and compares three training-set strategies:

- **No Oversampling:** the original imbalanced training fold is used unchanged.
- **SMOTE:** synthetic samples are generated from the minority class treated as one global pool.
- **G-SMOTE:** synthetic samples are generated inside geometric regions formed from the minority class treated as one global pool.

Each resulting training set is evaluated using the same five classifiers, folds, dataset-specific hyperparameters, and test observations.

### Phase 2: Cluster-based oversampling

Phase 2 retains the Phase 1 configurations and adds **Cluster+SMOTE** and **Cluster+G-SMOTE**.

For these strategies, the minority observations in each training fold are first partitioned with K-means. The number of clusters is selected with an elbow-based procedure and limited according to the number of synthetic observations required.

A custom four-factor rule determines how many synthetic samples each cluster receives. The rule considers:

1. relative cluster size,
2. intra-cluster compactness,
3. distance from the majority class,
4. connectedness to the remaining minority clusters.

Singleton clusters are treated as outliers and receive no synthetic samples. SMOTE or G-SMOTE is then applied inside each cluster. The augmented training set is finally evaluated with the same KNN classifiers used in Phase 1.

## Classification and sampling methods

### KNN-family classifiers

- Classical KNN
- Modified KNN with distance-based voting
- Fuzzy KNN with hard neighbor membership, `m=2`
- Fuzzy KNN with distance-to-class-mean membership, `m=2`
- Radius-based KNN

The classification logic is implemented from scratch using Euclidean distance. Scikit-learn is used only for K-means clustering in Phase 2.

### Standardization

Features are standardized separately inside every external fold. The mean and sample standard deviation are calculated from the training partition and then applied to both the training and test partitions. This prevents information from the test fold from leaking into training.

### SMOTE implementation

The implementation used in this project randomly selects two minority observations and places a synthetic observation at a random point on the line segment between them:

`x_new = lambda * x_1 + (1 - lambda) * x_2`, where `lambda ~ U(0,1)`.

The procedure is repeated until the minority and majority classes contain the same number of training observations.

### G-SMOTE implementation

G-SMOTE generates a synthetic observation inside a hypersphere rather than along a single line segment. For a selected minority center, the radius is determined using a minority neighbor and the nearest observed majority point. A random direction and a dimension-adjusted random radius are then used to draw a point inside the sphere.

The radius restriction prevents generated samples from extending beyond the nearest observed majority point, but it should not be interpreted as a mathematical guarantee about the unknown decision boundary.

### Hyperparameters

Candidate neighbor counts are `k=1,...,10`, while candidate radius values range from 0.5 to 5.0 in increments of 0.5. The selected dataset-specific values are stored in `BEST_PARAMS` and reused unchanged across all five training-set strategies. The selection procedure remains in the source code as a commented reproducibility reference.

## Evaluation metrics

The minority class is treated as the positive class.

- **Error rate:** proportion of all incorrect predictions.
- **Geometric Mean:** `sqrt(Sensitivity * Specificity)`; high only when both classes are recognized successfully.
- **F1 score:** harmonic mean of minority-class precision and recall.

The metrics are averaged across five stratified folds.

## Results

The following table summarizes the averages reported across the five classifiers and ten datasets.

| Training-set strategy | Error | G-Mean | F1 |
|---|---:|---:|---:|
| No Oversampling | **0.1326** | 0.7653 | 0.6770 |
| SMOTE | 0.1465 | **0.8214** | 0.6946 |
| G-SMOTE | 0.1409 | 0.8057 | 0.6966 |
| Cluster+SMOTE | 0.1443 | 0.8158 | 0.7016 |
| Cluster+G-SMOTE | **0.1375¹** | 0.8115 | **0.7046** |

¹ Lowest average error among the oversampling strategies. The no-oversampling baseline has the lowest overall error, but also the weakest average G-Mean and F1 score.

### Main findings

- Oversampling improves the imbalance-sensitive metrics compared with the no-oversampling baseline.
- SMOTE produces the highest average G-Mean, indicating the strongest overall balance between minority- and majority-class recognition.
- Cluster+G-SMOTE produces the highest average F1 score and the lowest error among the oversampling strategies.
- Cluster-based sampling does not outperform global oversampling on every dataset.
- Ecoli, Haberman, and Parkinson show favorable F1 behavior with Cluster+G-SMOTE.
- Large and highly imbalanced datasets such as CTO, PageBlock, and Yeast often favor standard SMOTE in G-Mean.
- PageBlock and Yeast demonstrate that increasing G-Mean does not necessarily increase F1; recovering more minority observations can also create additional false positives.
- The preferred strategy therefore depends on the evaluation objective and the local structure of the dataset.

The central conclusion is not that clustering is universally better. Global SMOTE is strongest when balanced recognition of both classes is the priority, while Cluster+G-SMOTE is strongest on average when minority-class F1 is prioritized.

## Datasets

Each workbook contains a `Description` sheet and a dataset sheet used by the program.

| Dataset | Source | Samples | Features | Class split (0 / 1) | Imbalance ratio |
|---|---|---:|---:|---:|---:|
| BCWD | [UCI ID 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 30 | 212 / 357 | 1.68x |
| CTO | [UCI ID 193](https://archive.ics.uci.edu/dataset/193/cardiotocography) | 2,126 | 21 | 134 / 1,992 | 14.87x |
| Ecoli | [UCI ID 39](https://archive.ics.uci.edu/dataset/39/ecoli) | 336 | 7 | 77 / 259 | 3.36x |
| Glass | [UCI ID 42](https://archive.ics.uci.edu/dataset/42/glass+identification) | 214 | 9 | 46 / 168 | 3.65x |
| Haberman | [UCI ID 43](https://archive.ics.uci.edu/dataset/43/haberman+s+survival) | 306 | 3 | 81 / 225 | 2.78x |
| Heart | [UCI ID 45](https://archive.ics.uci.edu/dataset/45/heart+disease) | 297 | 13 | 83 / 214 | 2.58x |
| PageBlock | [UCI ID 78](https://archive.ics.uci.edu/dataset/78/page+blocks+classification) | 5,473 | 10 | 560 / 4,913 | 8.77x |
| Parkinson | [UCI ID 174](https://archive.ics.uci.edu/dataset/174/parkinsons) | 195 | 22 | 48 / 147 | 3.06x |
| Pima | [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 768 | 8 | 500 / 268 | 1.87x |
| Yeast | [UCI ID 110](https://archive.ics.uci.edu/dataset/110/yeast) | 1,484 | 8 | 163 / 1,321 | 8.10x |

The original multiclass targets in CTO, Ecoli, Glass, Heart, PageBlock, and Yeast were converted into binary targets for this study. The exact mappings are documented in the workbook description sheets.

## Repository structure

```text
.
├── data/              # Ten Excel datasets
├── docs/report.pdf    # Full methodology and detailed results
├── src/main.py        # Complete experimental pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

## Running the project

Python 3.10 or later is recommended.

```bash
git clone https://github.com/emrebasaran1/imbalance-learning-clustering-classification.git
cd imbalance-learning-clustering-classification
pip install -r requirements.txt
python src/main.py
```

The implementation relies heavily on pure-Python distance calculations, so CTO, PageBlock, and Yeast can take considerably longer than the smaller datasets. The current script prints dataset progress and execution time. Detailed aggregate and dataset-level results are documented in the report.

## Report

The complete methodology, formulas, parameter table, aggregate findings, and dataset-level Phase 1 and Phase 2 results are available in [`docs/report.pdf`](docs/report.pdf).

## Reproducibility notes

- Random operations use the fixed seed `42`.
- Oversampling is performed only on training folds.
- Test folds preserve their original class distributions.
- Standardization parameters are calculated only from training observations.
- The same folds and classifier hyperparameters are used across the compared training-set strategies.
- Dataset-specific hyperparameters are stored directly in the source file.
- Results may depend on Python and dependency versions because the requirements specify minimum versions rather than a fully locked environment.

## License and data attribution

The source code and project documentation are released under the [MIT License](LICENSE).

The included datasets originate from the sources linked above. Their original terms and attribution requirements remain applicable; the repository's MIT License does not replace the licenses or usage conditions of the source datasets.
