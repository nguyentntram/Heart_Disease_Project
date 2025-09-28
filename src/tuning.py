from typing import Dict, Any
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def tune_random_forest(estimator: RandomForestClassifier, X, y, random_state: int = 42) -> GridSearchCV:
    param_grid = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(estimator, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    grid.fit(X, y)
    return grid

def tune_svm(estimator: SVC, X, y, random_state: int = 42) -> GridSearchCV:
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.001],
        "kernel": ["rbf"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(estimator, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    grid.fit(X, y)
    return grid