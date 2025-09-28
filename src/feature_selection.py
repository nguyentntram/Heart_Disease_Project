import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def chi2_ranking(X_mm: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    chi2_vals, p_vals = chi2(X_mm, y)
    df = pd.DataFrame({
        "feature": feature_names,
        "chi2": chi2_vals,
        "p_value": p_vals
    }).sort_values("chi2", ascending=False)
    return df

def rfe_ranking(estimator, X: np.ndarray, y: np.ndarray, feature_names: list, n_features_to_select: int = 10):
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "rank": selector.ranking_,
        "selected": selector.support_
    }).sort_values(["selected", "rank"], ascending=[False, True])
    return selector, ranking

def rf_importance(X: np.ndarray, y: np.ndarray, feature_names: list, random_state: int = 42):
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state)
    rf.fit(X, y)
    importances = rf.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return rf, df