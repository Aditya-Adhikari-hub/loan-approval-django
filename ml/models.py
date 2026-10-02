from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .preprocess import build_preprocessor


def get_candidates() -> dict:
    """name -> (estimator, param_grid)"""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5, 8],
                "clf__min_samples_leaf": [3, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                          verbosity=0, n_jobs=1),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    }


def make_pipeline(estimator, use_smote: bool = False):
    """preprocessor -> (optional SMOTE) -> classifier.
    SMOTE inside the pipeline only touches training folds, never validation/test."""
    if use_smote:
        return ImbPipeline([
            ("pre", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", estimator),
        ])
    return Pipeline([("pre", build_preprocessor()), ("clf", estimator)])