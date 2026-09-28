"""Baseline normal-vs-pneumonia classifier: logistic regression on flattened pixels.

Protocol
  - train: fit the model
  - val:   choose C (by ROC AUC) and the decision threshold (by Youden's J);
           val images that are exact duplicates of train images are excluded
  - test:  evaluated once, with the chosen C and threshold

Usage:
    python -m src.baseline
"""
import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.download_data import DEST as NPZ_PATH

C_GRID = [0.001, 0.01, 0.1, 1.0]
SEED = 0


def load_splits(npz_path=NPZ_PATH):
    """Returns {split: (X, y, images)} with X = flattened pixels scaled to [0, 1], y = 1 for pneumonia."""
    with np.load(npz_path, allow_pickle=False) as npz:
        out = {}
        for split in ("train", "val", "test"):
            images = npz[f"{split}_images"]
            X = images.reshape(len(images), -1).astype(np.float32) / 255.0
            out[split] = (X, npz[f"{split}_labels"].ravel().astype(int), images)
    return out


def not_in(images, reference_images):
    """Boolean mask: True where an image has no exact duplicate in reference_images."""
    ref = {hashlib.sha1(im.tobytes()).hexdigest() for im in reference_images}
    return np.array([hashlib.sha1(im.tobytes()).hexdigest() not in ref for im in images])


def make_model(C):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def select_model(data):
    """Fit one model per C on train; pick C by val ROC AUC, then the threshold by val Youden's J."""
    X_tr, y_tr, img_tr = data["train"]
    X_va, y_va, img_va = data["val"]
    keep = not_in(img_va, img_tr)
    X_va, y_va = X_va[keep], y_va[keep]

    search = []
    for C in C_GRID:
        model = make_model(C).fit(X_tr, y_tr)
        search.append({"C": C, "val_auc": roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]), "model": model})
    best = max(search, key=lambda r: r["val_auc"])

    fpr, tpr, thresholds = roc_curve(y_va, best["model"].predict_proba(X_va)[:, 1])
    j = np.argmax(tpr - fpr)
    selection = {
        "C": best["C"],
        "threshold": float(thresholds[j]),
        "val_auc": best["val_auc"],
        "val_sensitivity": tpr[j],
        "val_specificity": 1 - fpr[j],
        "n_val_used": int(keep.sum()),
        "n_val_dropped_dup": int((~keep).sum()),
    }
    search = [{k: v for k, v in r.items() if k != "model"} for r in search]
    return best["model"], selection, search


def evaluate(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n": len(y_true),
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity": tp / (tp + fn),      # recall for pneumonia
        "specificity": tn / (tn + fp),      # recall for normal
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "roc_auc": roc_auc_score(y_true, prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def save_metrics(selection, y_test, p_test):
    """Write test metrics (plus the always-pneumonia reference) to output/model_metrics.csv."""
    import pandas as pd

    from src.build_metadata import ROOT

    path = ROOT / "output" / "model_metrics.csv"
    pd.DataFrame([
        {"split": "test", "C": selection["C"], **evaluate(y_test, p_test, selection["threshold"])},
        {"split": "test (always pneumonia)", "C": None, **evaluate(y_test, np.ones(len(y_test)), 0.5)},
    ]).to_csv(path, index=False, float_format="%.4f")
    return path


if __name__ == "__main__":
    data = load_splits()
    model, selection, search = select_model(data)
    X_te, y_te, _ = data["test"]
    p_te = model.predict_proba(X_te)[:, 1]
    result = evaluate(y_te, p_te, selection["threshold"])
    print("C search (val AUC):", {r["C"]: round(r["val_auc"], 4) for r in search})
    print("Selected:", selection)
    print("Test:", {k: round(float(v), 4) for k, v in result.items()})
    print("Wrote", save_metrics(selection, y_te, p_te))
