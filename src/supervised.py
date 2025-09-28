from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models(random_state: int = 42) -> Dict[str, Any]:
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight=None, random_state=random_state),
        "dtree": DecisionTreeClassifier(random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "svm": SVC(kernel="rbf", probability=True, random_state=random_state),
    }
    return models