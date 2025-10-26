# -*- coding: utf-8 -*-
"""
Churn baseline (no argparse, no interactive input)
- Reads Excel file, runs preprocessing, trains Logistic Regression & Gradient Boosting
- Compatible with old/new scikit-learn (sparse vs sparse_output)
- Saves ROC curves -> 'roc_curves.png'
- Saves Precision–Recall curves -> 'pr_curves.png'
- Saves holdout predictions -> 'predictions_holdout.csv'
"""

import numpy as np
import pandas as pd

from packaging.version import Version
from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from ydata_profiling import ProfileReport

from sklearn.calibration import CalibrationDisplay

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from pathlib import Path


# ---------- SETTINGS ----------
EXCEL_PATH = "Test 1 - Tabela 1.xlsx"   # change if needed
TARGET = "CHURN"
ID_CANDIDATES = ("ID", "SUBSCRIB")
TEST_SIZE = 0.25
RANDOM_STATE = 42
MAX_ITER_LR = 2000

REPORT_NAME = "churn_profile_report.html"

# ---------- HELPERS ----------
def to01(x):
    """Map various truthy/falsey strings/numbers to {0,1}."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"1", "y", "yes", "true", "t"}: return 1
    if s in {"0", "n", "no", "false", "f"}: return 0
    try:
        v = float(s); return 1 if v >= 0.5 else 0
    except Exception:
        return np.nan

def make_onehot():
    """Return OneHotEncoder with correct parameter depending on sklearn version."""
    v = Version(skl_version)
    if v < Version("1.2.0"):
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)

def pick_id_cols(columns):
    """Find ID-like columns to exclude from features."""
    cols = []
    for c in columns:
        cu = str(c).upper()
        if any(tok in cu for tok in ID_CANDIDATES):
            cols.append(c)
    return cols


# ---------- MAIN ----------
def main():
    print("=== Running churn baseline ===")

    path = Path(EXCEL_PATH).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Load
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    # Target
    df[TARGET] = df[TARGET].apply(to01).astype("float").astype("Int64")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)




    # Features
    id_cols = pick_id_cols(df.columns)
    feature_cols = [c for c in df.columns if c not in id_cols + [TARGET]]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    print(f"[INFO] Records: {len(df)}")
    print(f"[INFO] Features: {len(feature_cols)} (num={len(num_cols)}, cat={len(cat_cols)})")
    if id_cols:
        print(f"[INFO] Excluding ID columns: {id_cols}")

    # 1. Check for near-perfect correlation with CHURN
    corrs = df.corr(numeric_only=True)["CHURN"].abs().sort_values(ascending=False)
    print(corrs.head(10))

    print(df["CHURN"].value_counts(normalize=True))

    # After df = pd.read_excel(path)
    missing_ratio = df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.4].index.tolist()

    if cols_to_drop:
        print(f"[INFO] Dropping columns with >40% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    id_cols = pick_id_cols(df.columns)
    feature_cols = [c for c in df.columns if c not in id_cols + [TARGET]]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]


    # ydata profiling
    # === RUN PROFILE ===
    profile = ProfileReport(
        df,
        title="Churn Dataset Profiling Report",
        explorative=True,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
        },
    )

    # === SAVE REPORT ===
    out_path = Path(REPORT_NAME).resolve()
    profile.to_file(out_path)
    print(f"[OK] Profiling report saved to: {out_path}")



    X = df[feature_cols].copy()
    y = df[TARGET].astype(int).values

    # Preprocess
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_onehot())
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=MAX_ITER_LR),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    # Train & evaluate
    all_probs = {}
    preds_to_save = {}

    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocess), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        ap = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  ROC AUC: {auc:.4f}  AP: {ap:.4f}")

        if y_proba is not None:
            all_probs[name] = y_proba
        preds_to_save[name] = (y_pred, y_proba)

    # ROC curves
    plt.figure(figsize=(7, 6))
    for name, y_proba in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – Holdout Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_roc = Path("roc_curves.png").resolve()
    plt.savefig(out_roc, dpi=150)
    print(f"[OK] ROC curves saved: {out_roc}")

    # Precision–Recall curves
    plt.figure(figsize=(7, 6))
    pos_rate = y_test.mean()
    for name, y_proba in all_probs.items():
        p, r, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(r, p, label=f"{name} (AP={ap:.3f})")
    # reference line: positive class prevalence (baseline classifier)
    plt.hlines(pos_rate, xmin=0, xmax=1, linestyles="--", linewidth=1, color="gray", label=f"Baseline={pos_rate:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves – Holdout Set")
    plt.legend(loc="lower left")
    plt.tight_layout()
    out_pr = Path("pr_curves.png").resolve()
    plt.savefig(out_pr, dpi=150)
    print(f"[OK] PR curves saved: {out_pr}")

    # Choose which model's probabilities to visualize (use LR by default; fall back to first available)
    model_for_plots = "LogisticRegression" if "LogisticRegression" in all_probs else next(iter(all_probs.keys()))
    y_proba_plot = all_probs[model_for_plots]
    print(f"[INFO] Plotting diagnostics for: {model_for_plots}")

    # 1) Overlapped distributions of predicted probabilities by true class
    plt.figure(figsize=(7, 6))
    mask1 = (y_test == 1)
    mask0 = (y_test == 0)

    # Histogram + KDE style overlay
    bins = np.linspace(0, 1, 30)
    plt.hist(y_proba_plot[mask0], bins=bins, density=True, alpha=0.5, label="Actual=0")
    plt.hist(y_proba_plot[mask1], bins=bins, density=True, alpha=0.5, label="Actual=1")

    # Optional smooth lines (simple rolling mean on hist counts) to mimic KDE feel without seaborn
    # (kept minimal: matplotlib only)
    plt.xlabel("Predicted probability (positive class)")
    plt.ylabel("Density")
    plt.title(f"Predicted Probability Distributions by Class – {model_for_plots}")
    plt.legend()
    plt.tight_layout()
    out_dist = Path("proba_distributions.png").resolve()
    plt.savefig(out_dist, dpi=150)
    print(f"[OK] Probability distributions saved: {out_dist}")

    # 2) Calibration / reliability curve
    plt.figure(figsize=(7, 6))
    CalibrationDisplay.from_predictions(
        y_test, y_proba_plot, n_bins=10, strategy="quantile"
    )
    plt.plot([0, 1], [0, 1], "--", linewidth=1)  # perfect calibration line
    plt.title(f"Calibration Curve – {model_for_plots}")
    plt.tight_layout()
    out_cal = Path("calibration_curve.png").resolve()
    plt.savefig(out_cal, dpi=150)
    print(f"[OK] Calibration curve saved: {out_cal}")

    # 3) Box/violin style comparison of probs by class (matplotlib-only boxplot)
    plt.figure(figsize=(7, 6))
    plt.boxplot([y_proba_plot[mask0], y_proba_plot[mask1]],
                labels=["Actual=0", "Actual=1"], showfliers=False)
    plt.ylabel("Predicted probability (positive class)")
    plt.title(f"Predicted Probabilities by True Class – {model_for_plots}")
    plt.tight_layout()
    out_box = Path("proba_by_class.png").resolve()
    plt.savefig(out_box, dpi=150)
    print(f"[OK] Probability-by-class plot saved: {out_box}")

    # 4) Print Pearson correlation between y_test and probabilities
    pearson_corr = np.corrcoef(y_test, y_proba_plot)[0, 1]
    print(f"[INFO] Pearson correlation (y_test vs predicted prob): {pearson_corr:.4f}")

    for name, (y_pred, _y_proba) in preds_to_save.items():
        # 1) Confusion matrix (counts)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Confusion Matrix – {name}")
        plt.tight_layout()
        out_cm = Path(f"confusion_matrix_{name}.png").resolve()
        plt.savefig(out_cm, dpi=150)
        print(f"[OK] Confusion matrix saved: {out_cm}")

        # 2) Normalized confusion matrix (per true class)
        cm_norm = confusion_matrix(y_test, y_pred, labels=[0, 1], normalize="true")
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=[0, 1])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f")
        ax.set_title(f"Normalized Confusion Matrix – {name}")
        plt.tight_layout()
        out_cmn = Path(f"confusion_matrix_norm_{name}.png").resolve()
        plt.savefig(out_cmn, dpi=150)
        print(f"[OK] Normalized confusion matrix saved: {out_cmn}")

        # 3) Classification report in console
        print(f"\n[INFO] Classification report — {name}:\n"
              f"{classification_report(y_test, y_pred, digits=4)}")


    # Save predictions
    out_csv = Path("predictions_holdout.csv").resolve()
    out_df = pd.DataFrame({"y_true": y_test})
    if id_cols:
        for c in id_cols:
            if c in df.columns:
                ids = df.loc[X_test.index, c]
                out_df[c] = ids.values
                break
    if "LogisticRegression" in preds_to_save:
        out_df["y_pred_LR"] = preds_to_save["LogisticRegression"][0]
        out_df["y_prob_LR"] = preds_to_save["LogisticRegression"][1]
    if "GradientBoosting" in preds_to_save:
        out_df["y_pred_GB"] = preds_to_save["GradientBoosting"][0]
        out_df["y_prob_GB"] = preds_to_save["GradientBoosting"][1]
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Predictions saved: {out_csv}")


if __name__ == "__main__":
    main()
