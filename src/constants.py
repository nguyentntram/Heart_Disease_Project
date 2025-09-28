# Expected column names for the Cleveland-style Heart Disease dataset
TARGET_COL = "target"

NUMERIC_COLS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_COLS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

# Positive class label (adjust if your dataset encodes differently)
POSITIVE_LABEL = 1
RANDOM_STATE = 42