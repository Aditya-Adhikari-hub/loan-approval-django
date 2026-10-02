import json
from functools import lru_cache

import joblib
import pandas as pd

from ml.config import FEATURES, METRICS_PATH, PIPELINE_PATH


class ModelNotAvailable(Exception):
    pass


@lru_cache(maxsize=1)
def get_pipeline():
    try:
        return joblib.load(PIPELINE_PATH)
    except FileNotFoundError as exc:
        raise ModelNotAvailable(
            f"{PIPELINE_PATH.name} not found. Run `python -m ml.train` first."
        ) from exc


@lru_cache(maxsize=1)
def get_model_info() -> dict:
    try:
        return json.loads(METRICS_PATH.read_text())
    except FileNotFoundError:
        return {}


def _input_frame(age, occupation, education_level, marital_status, income, credit_score):
    return pd.DataFrame([{
        "age": age,
        "income": float(income),
        "credit_score": credit_score,
        "occupation": occupation,
        "education_level": education_level,
        "marital_status": marital_status,
    }])[FEATURES]


def predict_loan(age, occupation, education_level, marital_status, income, credit_score):
    """Return the predicted status and probabilities for one applicant."""
    pipe = get_pipeline()
    row = _input_frame(age, occupation, education_level, marital_status, income, credit_score)

    proba = pipe.predict_proba(row)[0]
    p_approved = float(proba[list(pipe.classes_).index(1)])
    p_denied = 1.0 - p_approved
    return {
        "status": "APPROVED" if p_approved >= 0.5 else "DENIED",
        "confidence": max(p_approved, p_denied) * 100,
        "approval_probability": p_approved * 100,
        "denial_probability": p_denied * 100,
    }


FEATURE_LABELS = {
    "age": "Age",
    "income": "Annual income",
    "credit_score": "Credit score",
    "occupation": "Occupation",
    "education_level": "Education",
    "marital_status": "Marital status",
}


def explain_prediction(age, occupation, education_level, marital_status, income,
                       credit_score, top_n=4, min_effect=0.05):
    """Explain ONE prediction: how much each input pushed the result up or down.

    XGBoost can return exact per-feature contributions (TreeSHAP) itself via
    pred_contribs, so no extra library is needed. Contributions are in log-odds
    of approval: positive = raised the approval chance, negative = lowered it.
    One-hot columns are added back together so each original input appears once.

    Returns [] if the deployed model is not an XGBoost model.
    """
    pipe = get_pipeline()
    clf = pipe.named_steps["clf"]
    if clf.__class__.__name__ != "XGBClassifier":
        return []

    import xgboost as xgb

    row = _input_frame(age, occupation, education_level, marital_status, income, credit_score)
    pre = pipe.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    contribs = clf.get_booster().predict(
        xgb.DMatrix(pre.transform(row)), pred_contribs=True
    )[0][:-1]  # last value is the bias term

    totals = {f: 0.0 for f in FEATURES}
    for name, value in zip(names, contribs):
        owner = next(f for f in FEATURES if name == f or name.startswith(f + "_"))
        totals[owner] += float(value)

    shown = {
        "age": f"{age} years",
        "income": f"${float(income):,.0f}",
        "credit_score": str(credit_score),
        "occupation": occupation,
        "education_level": education_level,
        "marital_status": marital_status,
    }
    ranked = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)
    ranked = [(f, c) for f, c in ranked if abs(c) >= min_effect][:top_n]
    if not ranked:
        return []
    biggest = abs(ranked[0][1])

    def strength(c):
        c = abs(c)
        return "strongly" if c >= 1.0 else "moderately" if c >= 0.4 else "slightly"

    return [
        {
            "label": FEATURE_LABELS[f],
            "value": shown[f],
            "increases": c > 0,
            "strength": strength(c),
            "width": max(6, round(abs(c) / biggest * 100)),
        }
        for f, c in ranked
    ]