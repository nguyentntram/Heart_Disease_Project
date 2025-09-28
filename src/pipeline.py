from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from .constants import NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL, RANDOM_STATE

def build_preprocess_pipeline(scale_for_chi2: bool = False) -> ColumnTransformer:
    """Create a ColumnTransformer that imputes missing values and encodes features.

    If scale_for_chi2=True, use MinMaxScaler for numeric features to satisfy chi2 (non-negative requirement).
    Otherwise, use StandardScaler.
    """
    num_imputer = SimpleImputer(strategy="median")
    if scale_for_chi2:
        num_scaler = MinMaxScaler()
    else:
        num_scaler = StandardScaler()

    num_pipe = Pipeline(steps=[
        ("imputer", num_imputer),
        ("scaler", num_scaler),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CATEGORICAL_COLS),
    ])
    return pre

def attach_pca_to_pipeline(preprocessor: ColumnTransformer, n_components: float = 0.95) -> Pipeline:
    """Return a Pipeline that applies preprocessing and then PCA.

    n_components can be float (variance ratio) or int (number of components).
    """
    pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
    ])
    return pipe

def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def infer_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> Tuple[list, Dict[str, Any]]:
    """After fitting the preprocessor, attempt to recover feature names for downstream analysis."""
    # For OneHotEncoder feature names
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feat_names = list(preprocessor.named_transformers_["num"].get_feature_names_out(NUMERIC_COLS)) + cat_feature_names
    return feat_names, {"cat_feature_names": cat_feature_names}