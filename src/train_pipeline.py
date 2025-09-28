import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from joblib import dump

from .constants import TARGET_COL, RANDOM_STATE
from .pipeline import build_preprocess_pipeline, attach_pca_to_pipeline, split_X_y, infer_feature_names
from .supervised import get_models
from .feature_selection import chi2_ranking, rfe_ranking, rf_importance
from .tuning import tune_random_forest, tune_svm

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease.csv")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "evaluation_metrics.txt")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")

def save_metrics(text: str):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Place your CSV there.")

    df = pd.read_csv(DATA_PATH, na_values='?')

    df['target'] = df['target'].apply(lambda x: 1 if float(x) > 0 else 0)
    X, y = split_X_y(df)

    # --- Build preprocess pipeline (standard for modeling) ---
    pre = build_preprocess_pipeline(scale_for_chi2=False)
    pre.fit(X, y)
    X_tr = pre.transform(X)
    feat_names, _ = infer_feature_names(pre, X)

    # --- Feature Selection summaries ---
    pre_mm = build_preprocess_pipeline(scale_for_chi2=True)
    pre_mm.fit(X, y)
    X_mm = pre_mm.transform(X)

    chi2_df = chi2_ranking(X_mm, y.to_numpy(), feat_names)
    # RFE with LogisticRegression
    from sklearn.linear_model import LogisticRegression
    rfe_sel, rfe_df = rfe_ranking(LogisticRegression(max_iter=1000), X_tr, y.to_numpy(), feat_names, n_features_to_select=10)
    rf_model, rf_imp_df = rf_importance(X_tr, y.to_numpy(), feat_names)

    # --- Train/Test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    X_train_tr = pre.transform(X_train)
    X_test_tr  = pre.transform(X_test)

    # --- Baseline models ---
    models = get_models(random_state=RANDOM_STATE)
    metrics = []
    best = {"name": None, "auc": -1, "clf": None}

    for name, clf in models.items():
        clf.fit(X_train_tr, y_train)
        proba = clf.predict_proba(X_test_tr)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, proba)

        metrics.append({
            "model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc
        })

        if auc > best["auc"]:
            best = {"name": name, "auc": auc, "clf": clf}

    # --- Tuning best families (RF and SVM) ---
    tuned_summary = ""
    if "rf" in models:
        tuned_rf = tune_random_forest(models["rf"], X_train_tr, y_train)
        if tuned_rf.best_score_ > best["auc"]:
            best = {"name": "rf_tuned", "auc": tuned_rf.best_score_, "clf": tuned_rf.best_estimator_}
        tuned_summary += f"RF best params: {tuned_rf.best_params_}, best cv AUC: {tuned_rf.best_score_:.4f}\n"

    if "svm" in models:
        tuned_svm = tune_svm(models["svm"], X_train_tr, y_train)
        if tuned_svm.best_score_ > best["auc"]:
            best = {"name": "svm_tuned", "auc": tuned_svm.best_score_, "clf": tuned_svm.best_estimator_}
        tuned_summary += f"SVM best params: {tuned_svm.best_params_}, best cv AUC: {tuned_svm.best_score_:.4f}\n"

    # --- Persist full pipeline (preprocessor + best classifier) ---
    full_pipe = Pipeline(steps=[("pre", pre), ("clf", best["clf"])])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(full_pipe, MODEL_PATH)

    # --- Save evaluation report ---
    lines = []
    lines.append("=== BASELINE METRICS (test set) ===")
    for m in metrics:
        lines.append(f"{m['model']:>10s} | Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    lines.append("")
    lines.append("=== TUNING SUMMARY (CV AUC) ===")
    lines.append(tuned_summary.strip())
    lines.append("")
    lines.append(f"=== SELECTED BEST MODEL: {best['name']} (AUC={best['auc']:.4f}) ===")

    lines.append("")
    lines.append("=== FEATURE SELECTION SNAPSHOTS ===")
    lines.append("[Top 15 by Chi^2]")
    lines.append(chi2_df.head(15).to_string(index=False))
    lines.append("")
    lines.append("[Top 15 by RandomForest Importance]")
    lines.append(rf_imp_df.head(15).to_string(index=False))
    lines.append("")
    lines.append("[RFE (Top=selected=True)]")
    lines.append(rfe_df[rfe_df['selected']].to_string(index=False))

    save_metrics("\n".join(lines))
    print("Training complete. Metrics saved. Model exported.")

if __name__ == "__main__":
    main()