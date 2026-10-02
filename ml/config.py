from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "loan.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
PIPELINE_PATH = ARTIFACT_DIR / "loan_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

# Target: 1 = Approved, 0 = Denied
TARGET = "loan_status"
POSITIVE_LABEL = "Approved"

NUMERIC_FEATURES = ["age", "income", "credit_score"]
CATEGORICAL_FEATURES = ["occupation", "education_level", "marital_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# In the dataset but deliberately not used as a model input
EXCLUDED_FEATURES = ["gender"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SELECTION_METRIC = "f1_macro"