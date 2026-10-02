"""Run from the project root:  python -m ml.train"""
import json
import platform
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .config import (
    ARTIFACT_DIR, CATEGORICAL_FEATURES, CV_FOLDS, EXCLUDED_FEATURES,
    METRICS_PATH, NUMERIC_FEATURES, PIPELINE_PATH, RANDOM_STATE, SELECTION_METRIC,
)
from .models import get_candidates, make_pipeline
from .preprocess import load_data, make_xy, split_data, validate_data

warnings.filterwarnings("ignore")
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro", "roc_auc": "roc_auc"}


def evaluate(pipe, X, y) -> dict:
    """Positive class = Approved (1)."""
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _cv_summary(res, idx=None) -> dict:
    out = {}
    for m in SCORING:
        if idx is None:
            out[m], out[m + "_std"] = float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std())
        else:
            out[m], out[m + "_std"] = float(res[f"mean_test_{m}"][idx]), float(res[f"std_test_{m}"][idx])
    return out


def imbalance_experiment(X_train, y_train, cv) -> list:
    fixed = {
        "Logistic Regression": dict(C=1.0),
        "Random Forest": dict(n_estimators=200, max_depth=5, min_samples_leaf=3),
        "XGBoost": dict(n_estimators=150, max_depth=3, learning_rate=0.1),
    }
    rows = []
    for name, (est, _) in get_candidates().items():
        for strategy in ["none", "class_weight", "smote"]:
            model = est.set_params(**fixed[name])
            if strategy == "class_weight":
                if name == "XGBoost":
                    model.set_params(scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()))
                else:
                    model.set_params(class_weight="balanced")
            pipe = make_pipeline(model, use_smote=(strategy == "smote"))
            res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
            rows.append({"model": name, "strategy": strategy, **_cv_summary(res)})
    return rows


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    data_report = validate_data(df)
    X, y = make_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Rows: {len(df)} | train: {len(X_train)} | test: {len(X_test)}")

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline = {
        "accuracy": accuracy_score(y_test, dummy.predict(X_test)),
        "f1_macro": f1_score(y_test, dummy.predict(X_test), average="macro"),
    }
    print(f"Baseline accuracy={baseline['accuracy']:.3f}")

    print("\nImbalance experiment...")
    imb_rows = imbalance_experiment(X_train, y_train, cv)
    for r in imb_rows:
        print(f"  {r['model']:20s} {r['strategy']:13s} F1-macro={r['f1_macro']:.3f}  AUC={r['roc_auc']:.3f}")

    print("\nTuning models...")
    candidates = {}
    for name, (est, grid) in get_candidates().items():
        gs = GridSearchCV(make_pipeline(est), grid, scoring=SCORING,
                          refit=SELECTION_METRIC, cv=cv, n_jobs=1)
        gs.fit(X_train, y_train)
        best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        cv_scores = _cv_summary(gs.cv_results_, idx=gs.best_index_)
        test_scores = evaluate(gs.best_estimator_, X_test, y_test)
        candidates[name] = {"best_params": best_params, "cv": cv_scores, "test": test_scores}
        print(f"  {name:20s} CV F1-macro={cv_scores['f1_macro']:.3f} | "
              f"Test F1-macro={test_scores['f1_macro']:.3f} AUC={test_scores['roc_auc']:.3f}")

    # choose using CV only, never the test set
    selected = max(candidates, key=lambda n: candidates[n]["cv"][SELECTION_METRIC])
    print(f"\nSelected model: {selected}")

    # refit on all data for deployment
    final_pipe = make_pipeline(get_candidates()[selected][0])
    final_pipe.set_params(**{f"clf__{k}": v for k, v in candidates[selected]["best_params"].items()})
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, PIPELINE_PATH)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "config": {"random_state": RANDOM_STATE, "cv_folds": CV_FOLDS,
                   "selection_metric": SELECTION_METRIC,
                   "numeric_features": NUMERIC_FEATURES,
                   "categorical_features": CATEGORICAL_FEATURES,
                   "excluded_features": EXCLUDED_FEATURES, "positive_class": "Approved"},
        "data": {**data_report, "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "schema": {
            "numeric_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in NUMERIC_FEATURES},
            "categories": {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
        },
        "baseline_majority_class": baseline,
        "imbalance_experiment": imb_rows,
        "candidates": candidates,
        "selected_model": selected,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {PIPELINE_PATH.name} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()