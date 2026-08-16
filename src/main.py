#Emre Başaran - 2643740
#Final Project

import openpyxl
import random
import math
from pathlib import Path
from collections import defaultdict
import time

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEED = 42
random.seed(SEED)

#DOSYA OKUMA
def oku_excel(file_name, sheet_name):
    sheet = openpyxl.load_workbook(DATA_DIR / file_name, data_only=True)[sheet_name]
    features = []
    labels = []

    for i, row in enumerate(sheet.iter_rows(values_only=True)):
        if i == 0:
            continue

        row = list(row)
        if all(value is None for value in row):
            continue

        features.append(list(row[:-1]))
        labels.append(row[-1])
    return features, labels

features1,labels1  = oku_excel("BCWD.xlsx","BCWD")
features2,labels2  = oku_excel("CTO.xlsx","CTO")
features3,labels3  = oku_excel("Ecoli.xlsx","Ecoli")
features4,labels4  = oku_excel("Glass.xlsx","Glass")
features5,labels5  = oku_excel("Haberman.xlsx","Haberman")
features6,labels6  = oku_excel("Heart.xlsx","Heart")
features7,labels7  = oku_excel("PageBlock.xlsx","PageBlock")
features8,labels8  = oku_excel("Parkinson.xlsx","Parkinson")
features9,labels9  = oku_excel("Pima.xlsx","Pima")
features10,labels10 = oku_excel("Yeast.xlsx","Yeast")


datasets = [
    ("BCWD",      features1,  labels1),
    ("CTO",       features2,  labels2),
    ("Ecoli",     features3,  labels3),
    ("Glass",     features4,  labels4),
    ("Haberman",  features5,  labels5),
    ("Heart",     features6,  labels6),
    ("PageBlock", features7,  labels7),
    ("Parkinson", features8,  labels8),
    ("Pima",      features9,  labels9),
    ("Yeast",     features10, labels10)]

#STANDARDIZATION

def standardize_train_test(train_X, test_X):
    n_features = len(train_X[0])
    means = []
    stds  = []

    for j in range(n_features):
        column = [row[j] for row in train_X]
        mean   = sum(column) / len(column)
        std    = (sum((x - mean) ** 2 for x in column) / (len(column) - 1)) ** 0.5

        if std == 0:
            std = 1

        means.append(mean)
        stds.append(std)

    train_X_std = []
    for row in train_X:
        new_row = [(row[j] - means[j]) / stds[j] for j in range(n_features)]
        train_X_std.append(new_row)

    test_X_std = []
    for row in test_X:
        new_row = [(row[j] - means[j]) / stds[j] for j in range(n_features)]
        test_X_std.append(new_row)

    return train_X_std, test_X_std


#5-FOLD OLUSTURMA

def create_stratified_5_folds(features, labels, seed=SEED):
    rng = random.Random(seed)
    label_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        label_indices[label].append(idx)

    folds = [[] for _ in range(5)]

    for label in label_indices:
        indices = label_indices[label][:]
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % 5].append(idx)

    for fold in folds:
        rng.shuffle(fold)

    return folds

#TUM DATASETLER ICIN PARTITION OLUSTURMA

all_partitions = {}

for dataset_name, features, labels in datasets:
    folds = create_stratified_5_folds(features, labels, seed=SEED)
    dataset_partitions = []

    for fold_no in range(5):
        test_indices  = folds[fold_no]
        train_indices = []
        for i in range(5):
            if i != fold_no:
                train_indices.extend(folds[i])

        train_X = [features[i] for i in train_indices]
        train_y = [labels[i]   for i in train_indices]

        test_X  = [features[i] for i in test_indices]
        test_y  = [labels[i]   for i in test_indices]

        train_X_std, test_X_std = standardize_train_test(train_X, test_X)

        dataset_partitions.append({
            "fold":          fold_no + 1,
            "train_X":       train_X,
            "train_y":       train_y,
            "test_X":        test_X,
            "test_y":        test_y,
            "train_X_std":   train_X_std,
            "test_X_std":    test_X_std,
            "train_indices": train_indices,
            "test_indices":  test_indices})

    all_partitions[dataset_name] = dataset_partitions

#YARDIMCI FONKSIYONLAR

def compute_class_means(train_X, train_y):
    class_labels = list(set(train_y))
    n_features   = len(train_X[0])
    class_means  = {}

    for c in class_labels:
        indices  = [i for i in range(len(train_y)) if train_y[i] == c]
        mean_vec = [sum(train_X[i][j] for i in indices) / len(indices) for j in range(n_features)]
        class_means[c] = mean_vec

    return class_means


#KNN AlgoritmaLARI

# Classical KNN
def knn_predict(test_point, train_data, train_labels, k):
    distances = []
    for i in range(len(train_data)):
        d = math.dist(test_point, train_data[i])
        distances.append((d, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    votes = {}
    for d, label in k_nearest:
        votes[label] = votes.get(label, 0) + 1

    max_votes   = max(votes.values())
    tied_labels = [label for label, count in votes.items() if count == max_votes]

    if len(tied_labels) == 1:
        return tied_labels[0]

    for d, label in k_nearest:
        if label in tied_labels:
            return label

# Modified KNN
def modified_knn_predict(test_point, train_data, train_labels, k, exclude_idx=None):
    distances = []
    for i in range(len(train_data)):
        if i == exclude_idx:
            continue
        d = max(math.dist(test_point, train_data[i]), 1e-10)
        distances.append((d, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    d_min = k_nearest[0][0]
    d_max = k_nearest[-1][0]

    weighted_votes = {}
    for d, label in k_nearest:
        if d_max == d_min:
            w = 1.0
        elif d == d_min:
            w = 1.0
        else:
            w = (d_max - d) / (d_max - d_min)
        weighted_votes[label] = weighted_votes.get(label, 0) + w

    max_weight  = max(weighted_votes.values())
    tied_labels = [label for label, weight in weighted_votes.items() if weight == max_weight]

    if len(tied_labels) == 1:
        return tied_labels[0]

    for d, label in k_nearest:
        if label in tied_labels:
            return label

# Fuzzy KNN (mu in {0,1})
def fuzzy_knn_predict(test_point, train_data, train_labels, k, m=2, exclude_idx=None):
    distances = []
    for i in range(len(train_data)):
        if i == exclude_idx:
            continue
        d = max(math.dist(test_point, train_data[i]), 1e-10)
        distances.append((d, train_labels[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    class_labels = list(set(train_labels))
    memberships  = {c: 0.0 for c in class_labels}

    denominator = sum((1.0 / (d ** (2.0 / (m - 1)))) for d, label in k_nearest)

    for c in class_labels:
        numerator = 0.0
        for d, label in k_nearest:
            mu_ij = 1.0 if label == c else 0.0
            numerator += mu_ij * (1.0 / (d ** (2.0 / (m - 1))))
        memberships[c] = numerator / denominator

    return max(memberships, key=lambda c: memberships[c])

# Fuzzy KNN (mu in [0,1])
def fuzzy_knn_distance_predict(test_point, train_data, train_labels, k, class_means, m=2, exclude_idx=None):
    distances = []
    for i in range(len(train_data)):
        if i == exclude_idx:
            continue
        d = max(math.dist(test_point, train_data[i]), 1e-10)
        distances.append((d, train_labels[i], i))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    class_labels = list(class_means.keys())
    memberships  = {c: 0.0 for c in class_labels}
    denominator = sum((1.0 / (d ** (2.0 / (m - 1)))) for d, label, idx in k_nearest)

    for c in class_labels:
        numerator = 0.0
        for d, label, idx in k_nearest:
            d_to_means = {cl: max(math.dist(train_data[idx], class_means[cl]), 1e-10) for cl in class_labels}
            denom_mu   = sum(1.0 / d_to_means[cl] for cl in class_labels)
            mu_ij      = (1.0 / d_to_means[c]) / denom_mu
            numerator += mu_ij * (1.0 / (d ** (2.0 / (m - 1))))
        memberships[c] = numerator / denominator

    return max(memberships, key=lambda c: memberships[c])


# Radius KNN
def radius_knn_predict(test_point, train_data, train_labels, r, exclude_idx=None):
    neighbours = []
    for i in range(len(train_data)):
        if i == exclude_idx:
            continue
        d = max(math.dist(test_point, train_data[i]), 1e-10)
        if d <= r:
            neighbours.append((d, train_labels[i]))

    if len(neighbours) == 0:
        all_dist = []
        for i in range(len(train_data)):
            if i == exclude_idx:
                continue
            d = max(math.dist(test_point, train_data[i]), 1e-10)
            all_dist.append((d, train_labels[i]))
        all_dist.sort(key=lambda x: x[0])
        return all_dist[0][1]

    votes = {}
    for d, label in neighbours:
        votes[label] = votes.get(label, 0) + 1

    max_votes   = max(votes.values())
    tied_labels = [label for label, count in votes.items() if count == max_votes]

    if len(tied_labels) == 1:
        return tied_labels[0]

    neighbours.sort(key=lambda x: x[0])
    for d, label in neighbours:
        if label in tied_labels:
            return label

#SMOTE

def smote(train_X, train_y, seed=SEED):
    rng = random.Random(seed)
    counts = {}
    for label in train_y:
        counts[label] = counts.get(label, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    majority_label, majority_count = sorted_counts[0]
    minority_label, minority_count = sorted_counts[-1]
    n_to_generate = majority_count - minority_count

    if n_to_generate <= 0:
        return list(train_X), list(train_y)

    minority_points = [train_X[i] for i in range(len(train_y)) if train_y[i] == minority_label]
    n_features = len(minority_points[0])

    synthetic_X = []
    synthetic_y = []

    for _ in range(n_to_generate):
        idx1, idx2 = rng.sample(range(len(minority_points)), 2)
        x1 = minority_points[idx1]
        x2 = minority_points[idx2]

        lam = rng.random()
        new_point = [lam * x1[j] + (1 - lam) * x2[j] for j in range(n_features)]

        synthetic_X.append(new_point)
        synthetic_y.append(minority_label)

    new_train_X = list(train_X) + synthetic_X
    new_train_y = list(train_y) + synthetic_y

    return new_train_X, new_train_y

#G-SMOTE

def gsmote(train_X, train_y, k=5, seed=SEED):

    rng = random.Random(seed)

    counts = {}
    for label in train_y:
        counts[label] = counts.get(label, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    majority_label, majority_count = sorted_counts[0]
    minority_label, minority_count = sorted_counts[-1]

    n_to_generate = majority_count - minority_count

    if n_to_generate <= 0:
        return list(train_X), list(train_y)

    minority_pts = [train_X[i] for i in range(len(train_y)) if train_y[i] == minority_label]
    majority_pts = [train_X[i] for i in range(len(train_y)) if train_y[i] == majority_label]

    p = len(minority_pts[0])
    k_eff = min(k, len(minority_pts) - 1)
    if k_eff < 1:
        k_eff = 1

    minority_shuffled = minority_pts[:]
    rng.shuffle(minority_shuffled)

    synthetic_X = []
    synthetic_y = []

    for i in range(n_to_generate):
        xcenter = minority_shuffled[i % len(minority_shuffled)]
        xsurface = _gsmote_surface(xcenter, minority_pts, majority_pts, k_eff, rng)
        new_point = _gsmote_generate(xcenter, xsurface, p, rng)

        synthetic_X.append(new_point)
        synthetic_y.append(minority_label)
    return list(train_X) + synthetic_X, list(train_y) + synthetic_y


def _gsmote_surface(xcenter, minority_pts, majority_pts, k, rng):
    min_dists = []
    for pt in minority_pts:
        if pt is xcenter:
            continue
        min_dists.append((math.dist(xcenter, pt), pt))
    min_dists.sort(key=lambda x: x[0])
    min_neighbors = [pt for _, pt in min_dists[:k]]

    maj_dists = [(math.dist(xcenter, pt), pt) for pt in majority_pts]
    maj_dists.sort(key=lambda x: x[0])
    x_maj = maj_dists[0][1]

    if not min_neighbors:
        return x_maj

    x_min = rng.choice(min_neighbors)

    if math.dist(xcenter, x_min) <= math.dist(xcenter, x_maj):
        return x_min
    else:
        return x_maj

def _gsmote_generate(xcenter, xsurface, p, rng):

    R = math.dist(xcenter, xsurface)

    if R < 1e-12:
        return list(xcenter)

    v = [rng.gauss(0, 1) for _ in range(p)]
    n = math.sqrt(sum(x**2 for x in v))
    if n < 1e-12:
        v = [1.0] + [0.0] * (p - 1)
        n = 1.0

    r = rng.random() ** (1.0 / p)
    unit_pt = [r * (x / n) for x in v]
    return [xcenter[j] + R * unit_pt[j] for j in range(p)]

#BEST K and r calculation

# def _inner_find_best_k_r(itr, ival, ity, ivy, k_values, r_values):
#     cls    = list(set(ity))
#     cm     = compute_class_means(itr, ity)
#     cm_cls = list(cm.keys())
#     all_sorted = []
#     for tp in ival:
#         row = sorted(
#             (max(math.dist(tp, itr[j]), 1e-10), ity[j], j)
#             for j in range(len(itr))
#         )
#         all_sorted.append(row)

#     n_val     = len(ivy)
#     k_correct = {clf: {k: 0 for k in k_values}
#                  for clf in ["KNN", "ModKNN", "FuzzyV1", "FuzzyV2"]}
#     r_correct  = {r: 0 for r in r_values}

#     for i, true in enumerate(ivy):
#         row = all_sorted[i]

#         for k in k_values:
#             kn = row[:k]

#             v = {}
#             for d, l, _ in kn:
#                 v[l] = v.get(l, 0) + 1
#             mx   = max(v.values())
#             tied = [l for l, c in v.items() if c == mx]
#             pred = tied[0] if len(tied) == 1 else next(l for _, l, _ in kn if l in tied)
#             if pred == true:
#                 k_correct["KNN"][k] += 1

#             dmin, dmax = kn[0][0], kn[-1][0]
#             wv = {}
#             for di, l, _ in kn:
#                 w = 1.0 if dmax == dmin or di == dmin else (dmax - di) / (dmax - dmin)
#                 wv[l] = wv.get(l, 0) + w
#             mx   = max(wv.values())
#             tied = [l for l, w in wv.items() if w == mx]
#             pred = tied[0] if len(tied) == 1 else next(l for _, l, _ in kn if l in tied)
#             if pred == true:
#                 k_correct["ModKNN"][k] += 1

#             denom = sum(1 / (di ** 2) for di, _, _ in kn)
#             mem   = {c: sum((1.0 if l == c else 0.0) * (1 / (di ** 2))
#                             for di, l, _ in kn) / denom for c in cls}
#             if max(mem, key=lambda c: mem[c]) == true:
#                 k_correct["FuzzyV1"][k] += 1

#             denom2 = sum(1 / (di ** 2) for di, _, _ in kn)
#             mem2   = {}
#             for c in cm_cls:
#                 num = 0.0
#                 for di, l, idx in kn:
#                     dtm = {cl: max(math.dist(itr[idx], cm[cl]), 1e-10) for cl in cm_cls}
#                     dmu = sum(1 / dtm[cl] for cl in cm_cls)
#                     num += ((1 / dtm[c]) / dmu) * (1 / (di ** 2))
#                 mem2[c] = num / denom2
#             if max(mem2, key=lambda c: mem2[c]) == true:
#                 k_correct["FuzzyV2"][k] += 1

#         for r in r_values:
#             nb   = [(d, l) for d, l, _ in row if d <= r]
#             if not nb:
#                 nb = [(row[0][0], row[0][1])]
#             v    = {}
#             for d, l in nb:
#                 v[l] = v.get(l, 0) + 1
#             mx   = max(v.values())
#             tied = [l for l, c in v.items() if c == mx]
#             pred = tied[0] if len(tied) == 1 else next(l for _, l in sorted(nb) if l in tied)
#             if pred == true:
#                 r_correct[r] += 1

#     k_acc = {clf: {k: k_correct[clf][k] / n_val for k in k_values}
#              for clf in ["KNN", "ModKNN", "FuzzyV1", "FuzzyV2"]}
#     r_acc  = {r: r_correct[r] / n_val for r in r_values}
#     return k_acc, r_acc


# def find_best_hyperparams():
#     k_values = list(range(1, 11))
#     r_values = [round(0.5 * i, 1) for i in range(1, 11)]

#     best_params = {}

#     for dataset_name, features, labels in datasets:
#         print(f"{dataset_name}...", flush=True)
#         folds = create_stratified_5_folds(features, labels, seed=SEED)

#         k_scores = {clf: {k: [] for k in k_values}
#                     for clf in ["KNN", "ModKNN", "FuzzyV1", "FuzzyV2"]}
#         r_scores  = {r: [] for r in r_values}

#         for fold_no in range(5):
#             train_idx    = [i for fi in range(5) if fi != fold_no for i in folds[fi]]
#             train_X_full = [features[i] for i in train_idx]
#             train_y_full = [labels[i]   for i in train_idx]

#             split     = int(len(train_idx) * 0.8)
#             itr, ival = standardize_train_test(train_X_full[:split], train_X_full[split:])
#             ity       = train_y_full[:split]
#             ivy       = train_y_full[split:]

#             k_acc, r_acc = _inner_find_best_k_r(itr, ival, ity, ivy, k_values, r_values)

#             for clf in ["KNN", "ModKNN", "FuzzyV1", "FuzzyV2"]:
#                 for k in k_values:
#                     k_scores[clf][k].append(k_acc[clf][k])
#             for r in r_values:
#                 r_scores[r].append(r_acc[r])

#         ds_result = {}
#         for clf in ["KNN", "ModKNN", "FuzzyV1", "FuzzyV2"]:
#             ds_result[clf] = max(k_values, key=lambda k: sum(k_scores[clf][k]) / 5)
#         ds_result["RadiusKNN"] = max(r_values, key=lambda r: sum(r_scores[r]) / 5)

#         best_params[dataset_name] = ds_result
#         print(f"  {ds_result}", flush=True)

#     return best_params


# BEST_PARAMS = find_best_hyperparams()

BEST_PARAMS = {
    "BCWD":      {"KNN": 7,  "ModKNN": 9,  "FuzzyV1": 7,  "FuzzyV2": 10, "RadiusKNN": 4.0},
    "CTO":       {"KNN": 1,  "ModKNN": 1,  "FuzzyV1": 1,  "FuzzyV2": 8,  "RadiusKNN": 0.5},
    "Ecoli":     {"KNN": 1,  "ModKNN": 1,  "FuzzyV1": 1,  "FuzzyV2": 2,  "RadiusKNN": 0.5},
    "Glass":     {"KNN": 5,  "ModKNN": 8,  "FuzzyV1": 9,  "FuzzyV2": 1,  "RadiusKNN": 2.5},
    "Haberman":  {"KNN": 8,  "ModKNN": 10, "FuzzyV1": 9,  "FuzzyV2": 3,  "RadiusKNN": 5.0},
    "Heart":     {"KNN": 10, "ModKNN": 4,  "FuzzyV1": 8,  "FuzzyV2": 2,  "RadiusKNN": 3.0},
    "PageBlock": {"KNN": 4,  "ModKNN": 6,  "FuzzyV1": 5,  "FuzzyV2": 1,  "RadiusKNN": 0.5},
    "Parkinson": {"KNN": 3,  "ModKNN": 5,  "FuzzyV1": 3,  "FuzzyV2": 5,  "RadiusKNN": 2.0},
    "Pima":      {"KNN": 8,  "ModKNN": 10, "FuzzyV1": 9,  "FuzzyV2": 10, "RadiusKNN": 3.0},
    "Yeast":     {"KNN": 10, "ModKNN": 9,  "FuzzyV1": 10, "FuzzyV2": 4,  "RadiusKNN": 1.0}}

# 11) METRİK HESAPLAMA FORMÜLLERİ

def compute_metrics(y_true, y_pred, minority_label):
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == minority_label and p == minority_label)
    TN = sum(1 for t, p in zip(y_true, y_pred) if t != minority_label and p != minority_label)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t != minority_label and p == minority_label)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == minority_label and p != minority_label)
    n  = len(y_true)

    error = (FP + FN) / n

    sens  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    gmean = math.sqrt(sens * spec)

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return error, gmean, f1


# # 12) PHASE 1 ANA DONGUSU

def run_phase1():
    clf_names    = ["KNN", "ModKNN", "FuzzyV1", "FuzzyV2", "RadiusKNN"]
    method_names = ["No Oversampling", "SMOTE", "G-SMOTE"]
    metric_names = ["Error", "G-mean", "F1"]

    results = {ds: {clf: {} for clf in clf_names} for ds, _, _ in datasets}

    t0 = time.time()

    for dataset_name, features, labels in datasets:
        print(f"{dataset_name}...", flush=True)

        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        minority_label = min(counts, key=lambda l: counts[l])

        params = BEST_PARAMS[dataset_name]

        for fold_no in range(5):
            part    = all_partitions[dataset_name][fold_no]
            train_X = part["train_X_std"]
            train_y = part["train_y"]
            test_X  = part["test_X_std"]
            test_y  = part["test_y"]
            fold_seed = SEED + fold_no

            for method in method_names:
                if method == "No Oversampling":
                    tr_X, tr_y = train_X, train_y
                elif method == "SMOTE":
                    tr_X, tr_y = smote(train_X, train_y, seed=fold_seed)
                else:
                    tr_X, tr_y = gsmote(train_X, train_y, seed=fold_seed)

                cm = compute_class_means(tr_X, tr_y)

                for clf in clf_names:
                    if clf == "KNN":
                        k     = params["KNN"]
                        preds = [knn_predict(tp, tr_X, tr_y, k) for tp in test_X]
                    elif clf == "ModKNN":
                        k     = params["ModKNN"]
                        preds = [modified_knn_predict(tp, tr_X, tr_y, k) for tp in test_X]
                    elif clf == "FuzzyV1":
                        k     = params["FuzzyV1"]
                        preds = [fuzzy_knn_predict(tp, tr_X, tr_y, k) for tp in test_X]
                    elif clf == "FuzzyV2":
                        k     = params["FuzzyV2"]
                        preds = [fuzzy_knn_distance_predict(tp, tr_X, tr_y, k, cm) for tp in test_X]
                    else:
                        r     = params["RadiusKNN"]
                        preds = [radius_knn_predict(tp, tr_X, tr_y, r) for tp in test_X]

                    error, gmean, f1 = compute_metrics(test_y, preds, minority_label)

                    if method not in results[dataset_name][clf]:
                        results[dataset_name][clf][method] = [0.0, 0.0, 0.0]
                    results[dataset_name][clf][method][0] += error / 5
                    results[dataset_name][clf][method][1] += gmean / 5
                    results[dataset_name][clf][method][2] += f1    / 5

        print(f"  {time.time()-t0:.1f}s", flush=True)

run_phase1()


#PHASE 2 

EPS = 1e-12

def _centroid(points):
    p = len(points[0])
    n = len(points)
    return [sum(pt[j] for pt in points) / n for j in range(p)]


#CLISTERING
def cluster_minority(minority_pts, n_to_generate, seed=SEED):
    import numpy as np
    from sklearn.cluster import KMeans
    from kneed import KneeLocator

    n = len(minority_pts)
    if n <= 1:
        return [[list(p) for p in minority_pts]] if n else [[]]
    if n == 2:
        return [[list(minority_pts[0])], [list(minority_pts[1])]]

    k_max = math.isqrt(int(n_to_generate)) if n_to_generate > 0 else 2
    k_upper = min(k_max, n - 1)
    if k_upper < 2:
        k_upper = 2

    X = np.asarray(minority_pts, dtype=float)
    ks = list(range(2, k_upper + 1))
    if len(ks) == 1:
        best_k = 2
        labels = KMeans(n_clusters=2, random_state=seed, n_init=20).fit(X).labels_
    else:
        models = {k: KMeans(n_clusters=k, random_state=seed, n_init=20).fit(X) for k in ks}
        inertias = [models[k].inertia_ for k in ks]
        knee = KneeLocator(ks, inertias, curve="convex", direction="decreasing").elbow
        best_k = knee if knee is not None else 2
        labels = models[best_k].labels_

    clusters = [[] for _ in range(best_k)]
    for i, lab in enumerate(labels):
        clusters[lab].append(list(minority_pts[i]))
    return [c for c in clusters if c]


# WEIGHT FORMÜLLERİ
def cluster_weights(clusters, majority_pts):
    K = len(clusters)
    sizes = [len(c) for c in clusters]
    max_size = max(sizes) if sizes else 1
    centroids = [_centroid(c) if c else None for c in clusters]

    def avg_intra(c):
        m = len(c)
        if m < 2:
            return None
        tot = 0.0
        cnt = 0
        for i in range(m):
            for j in range(i + 1, m):
                tot += math.dist(c[i], c[j])
                cnt += 1
        return tot / cnt

    intra = [avg_intra(c) for c in clusters]
    valid = [v for v in intra if v is not None and v > EPS]
    min_intra = min(valid) if valid else EPS

    def avg_to_maj(c):
        if not c or not majority_pts:
            return 0.0
        return sum(sum(math.dist(pt, mp) for mp in majority_pts) / len(majority_pts)
                   for pt in c) / len(c)

    to_maj = [avg_to_maj(c) for c in clusters]
    max_to_maj = max(max(to_maj) if to_maj else EPS, EPS)

    def avg_to_others(idx):
        if K < 2 or centroids[idx] is None:
            return EPS
        ds = [math.dist(centroids[idx], centroids[j])
              for j in range(K) if j != idx and centroids[j] is not None]
        return sum(ds) / len(ds) if ds else EPS

    to_oth = [max(avg_to_others(i), EPS) for i in range(K)]
    min_inter = min(to_oth) if to_oth else EPS

    weights = []
    for i in range(K):
        if sizes[i] <= 1:
            weights.append(0.0)
            continue
        comp1 = sizes[i] / max_size
        ai = intra[i] if (intra[i] is not None and intra[i] > EPS) else min_intra
        comp2 = min_intra / ai
        comp3 = to_maj[i] / max_to_maj
        comp4 = min_inter / to_oth[i]
        weights.append(comp1 * comp2 * comp3 * comp4)
    return weights

def allocate_counts(weights, n_to_generate):
    s = sum(weights)
    if s <= 0:
        return [0] * len(weights)
    raw = [n_to_generate * w / s for w in weights]
    alloc = [int(math.floor(x)) for x in raw]
    rem = n_to_generate - sum(alloc)
    order = sorted(range(len(weights)), key=lambda i: (raw[i] - alloc[i]), reverse=True)
    i = 0
    while rem > 0 and order:
        alloc[order[i % len(order)]] += 1
        rem -= 1
        i += 1
    return alloc

def _smote_in_pool(pool, n, rng):
    out = []
    m = len(pool)
    if n <= 0 or m == 0:
        return out
    p = len(pool[0])
    if m == 1:
        return [list(pool[0]) for _ in range(n)]
    for _ in range(n):
        i, j = rng.sample(range(m), 2)
        lam = rng.random()
        out.append([lam * pool[i][t] + (1 - lam) * pool[j][t] for t in range(p)])
    return out


def _gsmote_in_pool(pool, majority_pts, n, k, rng):
    out = []
    m = len(pool)
    if n <= 0 or m == 0:
        return out
    p = len(pool[0])
    if m == 1:
        return [list(pool[0]) for _ in range(n)]
    k_eff = max(min(k, m - 1), 1)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    for i in range(n):
        xcenter = shuffled[i % m]
        xsurface = _gsmote_surface(xcenter, pool, majority_pts, k_eff, rng)
        out.append(_gsmote_generate(xcenter, xsurface, p, rng))
    return out


#Oversampling + Clustering formğlleri
def cluster_oversample(train_X, train_y, method, seed=SEED):
    rng = random.Random(seed)

    counts = {}
    for label in train_y:
        counts[label] = counts.get(label, 0) + 1
    sc = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    majority_label, majority_count = sc[0]
    minority_label, minority_count = sc[-1]

    n_to_generate = majority_count - minority_count
    if n_to_generate <= 0:
        return list(train_X), list(train_y)

    minority_pts = [train_X[i] for i in range(len(train_y)) if train_y[i] == minority_label]
    majority_pts = [train_X[i] for i in range(len(train_y)) if train_y[i] == majority_label]

    clusters = cluster_minority(minority_pts, n_to_generate, seed)
    weights = cluster_weights(clusters, majority_pts)

    if sum(weights) <= 0:
        fb = [len(c) if len(c) >= 2 else 0 for c in clusters]
        if sum(fb) <= 0:
            if method == "Cluster+SMOTE":
                syn = _smote_in_pool(minority_pts, n_to_generate, rng)
            else:
                syn = _gsmote_in_pool(minority_pts, majority_pts, n_to_generate, 5, rng)
            return list(train_X) + syn, list(train_y) + [minority_label] * len(syn)
        alloc = allocate_counts(fb, n_to_generate)
    else:
        alloc = allocate_counts(weights, n_to_generate)

    synthetic = []
    for c, a in zip(clusters, alloc):
        if a <= 0:
            continue
        if method == "Cluster+SMOTE":
            synthetic.extend(_smote_in_pool(c, a, rng))
        else:
            synthetic.extend(_gsmote_in_pool(c, majority_pts, a, 5, rng))

    return list(train_X) + synthetic, list(train_y) + [minority_label] * len(synthetic)


#Phase2 çalıştırma
def run_phase2(write_excel=False):

    clf_names = ["KNN", "ModKNN", "FuzzyV1", "FuzzyV2", "RadiusKNN"]
    method_names = ["No Oversampling", "SMOTE", "G-SMOTE", "Cluster+SMOTE", "Cluster+G-SMOTE"]
    metric_names = ["Error", "G-mean", "F1"]

    results = {ds: {clf: {} for clf in clf_names} for ds, _, _ in datasets}
    t0 = time.time()

    for dataset_name, features, labels in datasets:
        print(f"{dataset_name}...", flush=True)

        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        minority_label = min(counts, key=lambda l: counts[l])
        params = BEST_PARAMS[dataset_name]

        for fold_no in range(5):
            part = all_partitions[dataset_name][fold_no]
            train_X = part["train_X_std"]
            train_y = part["train_y"]
            test_X = part["test_X_std"]
            test_y = part["test_y"]
            fold_seed = SEED + fold_no

            for method in method_names:
                if method == "No Oversampling":
                    tr_X, tr_y = train_X, train_y
                elif method == "SMOTE":
                    tr_X, tr_y = smote(train_X, train_y, seed=fold_seed)
                elif method == "G-SMOTE":
                    tr_X, tr_y = gsmote(train_X, train_y, seed=fold_seed)
                else:
                    tr_X, tr_y = cluster_oversample(train_X, train_y, method, seed=fold_seed)

                cm = compute_class_means(tr_X, tr_y)

                for clf in clf_names:
                    if clf == "KNN":
                        k = params["KNN"]
                        preds = [knn_predict(tp, tr_X, tr_y, k) for tp in test_X]
                    elif clf == "ModKNN":
                        k = params["ModKNN"]
                        preds = [modified_knn_predict(tp, tr_X, tr_y, k) for tp in test_X]
                    elif clf == "FuzzyV1":
                        k = params["FuzzyV1"]
                        preds = [fuzzy_knn_predict(tp, tr_X, tr_y, k) for tp in test_X]
                    elif clf == "FuzzyV2":
                        k = params["FuzzyV2"]
                        preds = [fuzzy_knn_distance_predict(tp, tr_X, tr_y, k, cm) for tp in test_X]
                    else:
                        r = params["RadiusKNN"]
                        preds = [radius_knn_predict(tp, tr_X, tr_y, r) for tp in test_X]

                    error, gmean, f1 = compute_metrics(test_y, preds, minority_label)
                    if method not in results[dataset_name][clf]:
                        results[dataset_name][clf][method] = [0.0, 0.0, 0.0]
                    results[dataset_name][clf][method][0] += error / 5
                    results[dataset_name][clf][method][1] += gmean / 5
                    results[dataset_name][clf][method][2] += f1 / 5
        print(f"  {time.time() - t0:.1f}s", flush=True)
    print(f"\nToplam sure: {time.time() - t0:.1f}s")
    return results
run_phase2()
