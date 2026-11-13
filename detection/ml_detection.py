"""
Machine-learning fault detection.

A. Isolation Forest (unsupervised anomaly detection)
   Trained ONLY on healthy windows. The decision threshold is the 99th
   percentile of the healthy training anomaly scores (target ~1 % false
   alarms). Detects "something is different" without knowing fault types.
   Observed limitation: large multi-feature deviations (bearing, flow
   restriction, misalignment, imbalance) are detected well; faults that shift
   mainly ONE feature moderately (cavitation risk, overheating, seal leakage)
   are mostly missed - which is why rules and the classifier complement it.
   A 15-feature physically-normalised subset was tested and performed worse.
B. Random Forest (supervised fault classification, 8 classes)
   Trained on labelled windows; faulty windows with severity < 0.2 are left
   out of TRAINING because they are physically almost identical to healthy
   (label noise). All windows are kept for evaluation.

Leakage control
   * Train/test split by whole simulation runs (run_id), stratified by class:
     windows of one run never appear in both sets (adjacent windows of a
     time series are strongly correlated).
   * Multi-fault runs are test-only.
   * A second, completely fresh dataset (different seeds) is used as an
     independent validation set.
   * Only FEATURE_COLUMNS are used; ground-truth columns are metadata.

Performance on synthetic data does NOT establish industrial diagnostic
accuracy: the simulator and the features share the same modelling assumptions.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.feature_extraction import FEATURE_COLUMNS
from config.pump_config import project_path

MODEL_DIR = project_path("models/saved_models")
IF_PATH = MODEL_DIR / "isolation_forest.joblib"
RF_PATH = MODEL_DIR / "random_forest.joblib"
TRAIN_MIN_SEVERITY = 0.2
SEVERITY_BANDS = [0.0, 0.1, 0.3, 0.6, 1.0001]


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def run_class(run_id: str) -> str:
    return "multi" if run_id.startswith("multi_") else run_id.rsplit("_", 1)[0]


def split_by_run(df: pd.DataFrame, test_fraction: float = 0.33, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole runs, stratified by run class; multi-fault runs -> test."""
    rng = np.random.default_rng(seed)
    test_runs = []
    runs = pd.Series(df["run_id"].unique())
    for cls, group in runs.groupby(runs.map(run_class)):
        ids = list(group)
        if cls == "multi":
            test_runs += ids
            continue
        rng.shuffle(ids)
        test_runs += ids[:max(1, int(round(test_fraction * len(ids))))]
    is_test = df["run_id"].isin(test_runs)
    return df[~is_test].copy(), df[is_test].copy()


def _X(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=FEATURE_COLUMNS).astype(float)


# --------------------------------------------------------------------------
# Isolation Forest
# --------------------------------------------------------------------------
def train_isolation_forest(train: pd.DataFrame, seed: int = 0, quantile: float = 0.99) -> dict:
    healthy = train[train["fault_label"] == "healthy"]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("iforest", IsolationForest(n_estimators=300, random_state=seed))])
    pipe.fit(_X(healthy))
    scores = -pipe.score_samples(_X(healthy))            # higher = more anomalous
    return {"model": pipe, "threshold": float(np.quantile(scores, quantile)), "features": FEATURE_COLUMNS,
            "n_train": len(healthy), "quantile": quantile}


def anomaly_scores(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    return -bundle["model"].score_samples(df.reindex(columns=bundle["features"]).astype(float))


def evaluate_isolation_forest(bundle: dict, test: pd.DataFrame) -> dict:
    s = anomaly_scores(bundle, test)
    flag = s > bundle["threshold"]
    truth = (test["fault_label"] != "healthy").to_numpy()
    by_band = (pd.DataFrame({"sev": test["fault_severity"].to_numpy(), "flag": flag})[truth]
               .assign(band=lambda d: pd.cut(d.sev, SEVERITY_BANDS, right=False))
               .groupby("band", observed=True).flag.agg(["mean", "size"]))
    by_class = pd.DataFrame({"label": test["fault_label"].to_numpy(), "flag": flag}).groupby("label").flag.mean()
    return {"false_positive_rate": float(flag[~truth].mean()), "detection_rate": float(flag[truth].mean()),
            "roc_auc": float(roc_auc_score(truth, s)), "detection_by_severity": by_band,
            "flag_rate_by_class": by_class}


# --------------------------------------------------------------------------
# Random Forest
# --------------------------------------------------------------------------
def train_random_forest(train: pd.DataFrame, seed: int = 0, min_severity: float = TRAIN_MIN_SEVERITY) -> dict:
    use = train[(train["fault_label"] == "healthy") | (train["fault_severity"] >= min_severity)]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                                   random_state=seed, n_jobs=-1))])
    pipe.fit(_X(use), use["fault_label"])
    return {"model": pipe, "features": FEATURE_COLUMNS, "classes": list(pipe.classes_), "n_train": len(use),
            "train_min_severity": min_severity}


def evaluate_random_forest(bundle: dict, test: pd.DataFrame, min_severity: float = 0.0) -> dict:
    t = test[(test["fault_label"] == "healthy") | (test["fault_severity"] >= min_severity)]
    y, pred = t["fault_label"].to_numpy(), bundle["model"].predict(_X(t))
    labels = sorted(set(y) | set(pred))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    truth_faulty, pred_faulty = y != "healthy", pred != "healthy"
    multi = t["run_id"].str.startswith("multi_").to_numpy()
    single_acc = float((pred[~multi] == y[~multi]).mean()) if (~multi).any() else float("nan")
    mf = multi & truth_faulty
    multi_any = (float(np.mean([p in a.split("+") for p, a in zip(pred[mf], t["active_faults"].to_numpy()[mf])]))
                 if mf.any() else float("nan"))
    return {
        "windows": len(t),
        "single_fault_run_accuracy": single_acc,
        "multi_fault_any_active_match": multi_any,
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "false_positive_rate": float(pred_faulty[~truth_faulty].mean()) if (~truth_faulty).any() else 0.0,
        "detection_rate": float(pred_faulty[truth_faulty].mean()) if truth_faulty.any() else 0.0,
        "report": pd.DataFrame(classification_report(y, pred, labels=labels, output_dict=True,
                                                     zero_division=0)).T,
        "confusion": pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels),
        "predictions": pred,
    }


def feature_importance(bundle: dict, test: pd.DataFrame, seed: int = 0, top: int = 15) -> pd.DataFrame:
    rf = bundle["model"].named_steps["rf"]
    imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "impurity": rf.feature_importances_})
    t = test[(test["fault_label"] == "healthy") | (test["fault_severity"] >= TRAIN_MIN_SEVERITY)]
    perm = permutation_importance(bundle["model"], _X(t), t["fault_label"], n_repeats=5, random_state=seed,
                                  scoring="f1_macro", n_jobs=-1)
    imp["permutation"] = perm.importances_mean
    return imp.sort_values("permutation", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------
# Persistence and live inference
# --------------------------------------------------------------------------
def save_bundle(bundle: dict, path: Path, extra: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"saved_at": datetime.now().isoformat(timespec="seconds"), "sklearn_version": sklearn.__version__}
    joblib.dump({**bundle, **(extra or {}), "meta": meta}, path)
    return path


class MLDetector:
    """Loads the saved models and scores feature windows (dict or DataFrame)."""

    def __init__(self, if_path: Path = IF_PATH, rf_path: Path = RF_PATH):
        self.iforest = joblib.load(if_path)
        self.rf = joblib.load(rf_path)

    @staticmethod
    def available(if_path: Path = IF_PATH, rf_path: Path = RF_PATH) -> bool:
        return Path(if_path).exists() and Path(rf_path).exists()

    def predict(self, features) -> pd.DataFrame:
        df = pd.DataFrame([features]) if isinstance(features, dict) else features
        score = anomaly_scores(self.iforest, df)
        proba = self.rf["model"].predict_proba(df.reindex(columns=self.rf["features"]).astype(float))
        classes = np.array(self.rf["classes"])
        out = pd.DataFrame({"anomaly_score": score, "is_anomaly": score > self.iforest["threshold"],
                            "predicted_fault": classes[proba.argmax(1)], "confidence": proba.max(1)},
                           index=df.index)
        for i, c in enumerate(classes):
            out[f"p_{c}"] = proba[:, i]
        return out


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def _plots(conf: pd.DataFrame, imp: pd.DataFrame, path_cm: str, path_imp: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = conf.div(conf.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(conf)), conf.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(conf)), conf.index)
    for i in range(len(conf)):
        for j in range(len(conf)):
            ax.text(j, i, f"{conf.iat[i, j]}", ha="center", va="center", fontsize=9,
                    color="white" if norm.iat[i, j] > 0.5 else "black")
    ax.set(xlabel="Predicted", ylabel="True", title="Random Forest confusion matrix (held-out runs, all severities)")
    fig.tight_layout()
    fig.savefig(path_cm, dpi=90)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(imp))[::-1]
    ax.barh(y + 0.2, imp.permutation, height=0.4, label="permutation (Δ macro-F1, held-out runs)")
    ax.barh(y - 0.2, imp.impurity, height=0.4, label="impurity (Gini, training)")
    ax.set_yticks(y, imp.feature)
    ax.set(xlabel="importance", title="Random Forest feature importance (top 15)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_imp, dpi=90)
    plt.close(fig)


def train_and_evaluate(features_path: str = "data/processed/features.csv",
                       validation_path: str = "data/processed/features_validation.csv", seed: int = 0) -> dict:
    from analysis.feature_extraction import build_feature_dataset
    from detection.rule_based import evaluate as rule_evaluate

    data = pd.read_csv(features_path)
    if Path(validation_path).exists():
        val = pd.read_csv(validation_path)
    else:
        val = build_feature_dataset(runs_per_class=3, path=validation_path, seed=2000)
    train, test = split_by_run(data, seed=seed)
    assert set(train.run_id).isdisjoint(test.run_id), "run leakage"

    iforest = train_isolation_forest(train, seed)
    rf = train_random_forest(train, seed)
    results = {"split": {"train_runs": train.run_id.nunique(), "test_runs": test.run_id.nunique(),
                         "train_windows": len(train), "test_windows": len(test),
                         "validation_runs": val.run_id.nunique(), "validation_windows": len(val)}}
    for name, d in (("test", test), ("validation", val)):
        results[f"if_{name}"] = evaluate_isolation_forest(iforest, d)
        for sev in (0.0, 0.3):
            results[f"rf_{name}_sev{sev}"] = evaluate_random_forest(rf, d, sev)
            results[f"rules_{name}_sev{sev}"] = rule_evaluate(d, min_severity=sev)["summary"]
    imp = feature_importance(rf, test, seed)
    results["importance"] = imp
    save_bundle(iforest, IF_PATH, {"train_runs": sorted(train.run_id.unique())})
    save_bundle(rf, RF_PATH, {"train_runs": sorted(train.run_id.unique())})
    _plots(results["rf_test_sev0.0"]["confusion"], imp, "reports/ml_confusion_matrix.png",
           "reports/ml_feature_importance.png")

    scalar = {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
              for k, v in results.items() if isinstance(v, dict)}
    Path("reports/ml_metrics.json").write_text(json.dumps(scalar, indent=2))
    return results


if __name__ == "__main__":
    res = train_and_evaluate()
    r3 = lambda d: {k: round(v, 3) for k, v in d.items() if isinstance(v, (int, float))}
    print("Split:", res["split"])
    for name in ("test", "validation"):
        print(f"\n===== {name.upper()} =====")
        print("Isolation Forest:", r3(res[f"if_{name}"]))
        print(" detection by severity band:\n", res[f"if_{name}"]["detection_by_severity"].round(3).to_string())
        for sev in (0.0, 0.3):
            print(f"Random Forest sev>={sev}:", r3(res[f"rf_{name}_sev{sev}"]))
            print(f"Rule-based    sev>={sev}:", r3(res[f"rules_{name}_sev{sev}"]))
    print("\nPer-class (validation, sev>=0.3):")
    print(res["rf_validation_sev0.3"]["report"].round(3).to_string())
    print("\nConfusion (test, all severities):")
    print(res["rf_test_sev0.0"]["confusion"].to_string())
    print("\nTop features:\n", res["importance"].round(4).to_string())
