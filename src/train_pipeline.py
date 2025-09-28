import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from joblib import dump

# Relative imports because we run this as a module: `python -m src.train_pipeline`
from .constants import TARGET_COL, RANDOM_STATE
from .pipeline import build_preprocess_pipeline, split_X_y, infer_feature_names
from .supervised import get_models
from .feature_selection import chi2_ranking, rfe_ranking, rf_importance
from .tuning import tune_random_forest, tune_svm


# ---- paths ----
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data",   "heart_disease.csv")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_metrics.txt")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "final_model.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def save_text(path: str, text: str):
    """Save plain text to a file (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    # ---------- 0) Load and clean data ----------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Place your CSV there.")

    # Read CSV; treat "?" as missing (as in UCI Cleveland)
    df = pd.read_csv(DATA_PATH, na_values='?')

    # Convert target to binary: 0 = no disease, 1 = has disease (merge 1..4 -> 1)
    df[TARGET_COL] = df[TARGET_COL].apply(lambda x: 1 if float(x) > 0 else 0)

    X, y = split_X_y(df)

    # ---------- 1) Split before any fitting to avoid data leakage ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # ---------- 2) Preprocess pipeline (fit ONLY on train) ----------
    pre = build_preprocess_pipeline(scale_for_chi2=False)
    X_train_tr = pre.fit_transform(X_train, y_train)  # fit on train
    X_test_tr  = pre.transform(X_test)

    # Infer feature names based on the fitted preprocessor (for feature selection reports)
    feat_names, _ = infer_feature_names(pre, X_train)

    # ---------- 3) Feature Selection (fit ONLY on train) ----------
    # For chi2 we need non-negative features, so use MinMax scaling pipeline
    pre_mm = build_preprocess_pipeline(scale_for_chi2=True)
    X_train_mm = pre_mm.fit_transform(X_train, y_train)

    chi2_df = chi2_ranking(X_train_mm, y_train.to_numpy(), feat_names)

    # RFE with Logistic Regression
    from sklearn.linear_model import LogisticRegression
    rfe_sel, rfe_df = rfe_ranking(
        LogisticRegression(max_iter=1000),
        X_train_tr, y_train.to_numpy(),
        feat_names, n_features_to_select=10
    )

    # Random Forest feature importance
    rf_model, rf_imp_df = rf_importance(X_train_tr, y_train.to_numpy(), feat_names)

    # ---------- 4) Baseline supervised models (evaluate on TEST set) ----------
    models = get_models(random_state=RANDOM_STATE)
    metrics = []
    best = {"name": None, "auc": -1.0, "clf": None}

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
            "model": name,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc
        })

        if auc > best["auc"]:
            best = {"name": name, "auc": auc, "clf": clf}

    # ---------- 5) Hyperparameter tuning on train; compare by TEST AUC ----------
    tuned_summary = ""
    if "rf" in models:
        tuned_rf = tune_random_forest(models["rf"], X_train_tr, y_train)  # CV on train, refit=True
        rf_test_auc = roc_auc_score(y_test, tuned_rf.best_estimator_.predict_proba(X_test_tr)[:, 1])
        tuned_summary += f"RF best params: {tuned_rf.best_params_}, CV AUC: {tuned_rf.best_score_:.4f}, TEST AUC: {rf_test_auc:.4f}\n"
        if rf_test_auc > best["auc"]:
            best = {"name": "rf_tuned", "auc": rf_test_auc, "clf": tuned_rf.best_estimator_}

    if "svm" in models:
        tuned_svm = tune_svm(models["svm"], X_train_tr, y_train)  # CV on train, refit=True
        svm_test_auc = roc_auc_score(y_test, tuned_svm.best_estimator_.predict_proba(X_test_tr)[:, 1])
        tuned_summary += f"SVM best params: {tuned_svm.best_params_}, CV AUC: {tuned_svm.best_score_:.4f}, TEST AUC: {svm_test_auc:.4f}\n"
        if svm_test_auc > best["auc"]:
            best = {"name": "svm_tuned", "auc": svm_test_auc, "clf": tuned_svm.best_estimator_}

    # ---------- 6) Persist the full pipeline (fitted preprocessor + fitted best classifier) ----------
    # We store the fitted preprocessor and the already-fitted best classifier together.
    full_pipe = Pipeline(steps=[("pre", pre), ("clf", best["clf"])])
    dump(full_pipe, MODEL_PATH)

    # ---------- 7) Save evaluation report ----------
    lines = []
    lines.append("=== BASELINE METRICS (test set) ===")
    for m in metrics:
        lines.append(
            f"{m['model']:>10s} | "
            f"Acc={m['accuracy']:.4f}  Prec={m['precision']:.4f}  "
            f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}"
        )
    lines.append("")
    lines.append("=== TUNING SUMMARY ===")
    lines.append(tuned_summary.strip())
    lines.append("")
    lines.append(f"=== SELECTED BEST MODEL: {best['name']} (TEST AUC={best['auc']:.4f}) ===")
    lines.append("")
    lines.append("=== FEATURE SELECTION SNAPSHOTS (train only) ===")
    lines.append("[Top 15 by Chi^2]")
    lines.append(chi2_df.head(15).to_string(index=False))
    lines.append("")
    lines.append("[Top 15 by RandomForest Importance]")
    lines.append(rf_imp_df.head(15).to_string(index=False))
    lines.append("")
    lines.append("[RFE (Top=selected=True)]")
    lines.append(rfe_df[rfe_df['selected']].to_string(index=False))

    save_text(RESULTS_PATH, "\n".join(lines))

    # ---------- 8) Save artifacts (ROC curve, Confusion Matrix, and CSVs) ----------
    # Best model predictions on test set
    best_proba = best["clf"].predict_proba(X_test_tr)[:, 1]
    best_pred = (best_proba >= 0.5).astype(int)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, best_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, best_proba):.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    disp = ConfusionMatrixDisplay(cm)
    plt.figure()
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {best['name']}")
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Export Feature Selection CSVs (based on TRAIN)
    chi2_df.to_csv(os.path.join(RESULTS_DIR, "chi2_top.csv"), index=False)
    rf_imp_df.to_csv(os.path.join(RESULTS_DIR, "rf_feature_importance.csv"), index=False)
    rfe_df.to_csv(os.path.join(RESULTS_DIR, "rfe_ranking.csv"), index=False)

    print("Training complete. Metrics & artifacts saved. Model exported.")


if __name__ == "__main__":
    main()
