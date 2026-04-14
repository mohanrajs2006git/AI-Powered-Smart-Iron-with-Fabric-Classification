"""
╔══════════════════════════════════════════════════════════════════╗
║           SMART IRON — FABRIC TYPE CLASSIFIER                   ║
║           Full Training Pipeline with Best Accuracy             ║
╚══════════════════════════════════════════════════════════════════╝

Dataset  : smart_iron_dataset_corrected.xlsx
Features : Temperature_C, Motion_Variation, Static_Time_s
Target   : Fabric_Type (Cotton, Silk, Wool, Polyester, Anomaly)
Model    : Voting Ensemble (Extra Trees + Random Forest + SVM + GBM)
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.model_selection      import (train_test_split, StratifiedKFold,
                                           cross_val_score, GridSearchCV)
from sklearn.preprocessing        import LabelEncoder, RobustScaler
from sklearn.pipeline             import Pipeline
from sklearn.ensemble             import (RandomForestClassifier,
                                           ExtraTreesClassifier,
                                           GradientBoostingClassifier,
                                           VotingClassifier)
from sklearn.svm                  import SVC
from sklearn.neighbors            import KNeighborsClassifier
from sklearn.linear_model         import LogisticRegression
from sklearn.metrics              import (accuracy_score, f1_score,
                                           classification_report,
                                           confusion_matrix,
                                           ConfusionMatrixDisplay)
from sklearn.inspection           import permutation_importance


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    print("\n" + "═" * 62)
    print("  STEP 1 · LOADING DATA")
    print("═" * 62)

    df = pd.read_excel(filepath)

    print(f"  ✔ Loaded  : {filepath}")
    print(f"  ✔ Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  ✔ Columns : {df.columns.tolist()}")
    print(f"\n  Class Distribution:")
    for label, count in df["Fabric_Type"].value_counts().items():
        bar = "█" * (count // 5)
        print(f"    {label:<12} {count:>4}  {bar}")

    missing = df.isnull().sum().sum()
    print(f"\n  ✔ Missing Values : {missing}")
    if missing > 0:
        print("  ⚠ Filling missing values with column medians...")
        df.fillna(df.median(numeric_only=True), inplace=True)

    print(f"\n  Summary Statistics:")
    print(df.describe().round(4).to_string())
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═" * 62)
    print("  STEP 2 · FEATURE ENGINEERING")
    print("═" * 62)

    d = df.copy()
    eps = 1e-9   # prevent divide-by-zero

    # Physics-inspired interactions
    d["Temp_Motion_Ratio"]    = d["Temperature_C"]    / (d["Motion_Variation"] + eps)
    d["Temp_Static_Product"]  = d["Temperature_C"]    *  d["Static_Time_s"]
    d["Motion_Static_Ratio"]  = d["Motion_Variation"] / (d["Static_Time_s"]   + eps)
    d["Temp_Motion_Product"]  = d["Temperature_C"]    *  d["Motion_Variation"]

    # Log transforms (compress right-skewed sensor data)
    d["Motion_log"]           = np.log1p(d["Motion_Variation"])
    d["Static_log"]           = np.log1p(d["Static_Time_s"])

    # Polynomial
    d["Temp_squared"]         = d["Temperature_C"] ** 2
    d["Motion_squared"]       = d["Motion_Variation"] ** 2

    # Domain-specific: heat exposure index (temp × time iron is still)
    d["Heat_Exposure_Index"]  = d["Temperature_C"] * d["Static_Time_s"] / (d["Motion_Variation"] + eps)

    new_features = [c for c in d.columns if c not in df.columns]
    print(f"  ✔ Original features  : {len(df.columns) - 1}")
    print(f"  ✔ Engineered features: {len(new_features)}")
    print(f"  ✔ Total features     : {len(d.columns) - 1}")
    print(f"  New: {new_features}")
    return d


# ─────────────────────────────────────────────────────────────────
# STEP 3 — ENCODE & SPLIT
# ─────────────────────────────────────────────────────────────────
def encode_and_split(df: pd.DataFrame, target: str = "Fabric_Type",
                     test_size: float = 0.2, random_state: int = 42):
    print("\n" + "═" * 62)
    print("  STEP 3 · ENCODE & TRAIN/TEST SPLIT")
    print("═" * 62)

    le = LabelEncoder()
    X  = df.drop(columns=[target])
    y  = le.fit_transform(df[target])

    print(f"  Label mapping:")
    for cls, idx in zip(le.classes_, le.transform(le.classes_)):
        print(f"    {idx} → {cls}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"\n  ✔ Train set : {X_train.shape[0]} samples")
    print(f"  ✔ Test set  : {X_test.shape[0]} samples")
    print(f"  ✔ Features  : {X_train.shape[1]}")

    # Class balance check
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n  Train class distribution:")
    for u, c in zip(unique, counts):
        print(f"    {le.inverse_transform([u])[0]:<12} → {c} samples")

    return X_train, X_test, y_train, y_test, le


# ─────────────────────────────────────────────────────────────────
# STEP 4 — BUILD MODELS
# ─────────────────────────────────────────────────────────────────
def build_models() -> dict:
    print("\n" + "═" * 62)
    print("  STEP 4 · BUILDING MODELS")
    print("═" * 62)

    scaler = RobustScaler()

    models = {

        "Extra Trees": Pipeline([
            ("scaler", RobustScaler()),
            ("clf",    ExtraTreesClassifier(
                n_estimators   = 500,
                max_depth      = None,
                min_samples_split = 2,
                min_samples_leaf  = 1,
                max_features   = "sqrt",
                class_weight   = "balanced",
                random_state   = 42,
                n_jobs         = -1
            ))
        ]),

        "Random Forest": Pipeline([
            ("scaler", RobustScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators   = 500,
                max_depth      = None,
                min_samples_split = 2,
                min_samples_leaf  = 1,
                max_features   = "sqrt",
                class_weight   = "balanced",
                random_state   = 42,
                n_jobs         = -1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("scaler", RobustScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators   = 300,
                learning_rate  = 0.05,
                max_depth      = 5,
                subsample      = 0.8,
                min_samples_split = 4,
                random_state   = 42
            ))
        ]),

        "SVM (RBF)": Pipeline([
            ("scaler", RobustScaler()),
            ("clf",    SVC(
                C              = 10,
                kernel         = "rbf",
                gamma          = "scale",
                class_weight   = "balanced",
                probability    = True,
                random_state   = 42
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", RobustScaler()),
            ("clf",    KNeighborsClassifier(
                n_neighbors    = 5,
                weights        = "distance",
                metric         = "euclidean"
            ))
        ]),

        "Logistic Regression": Pipeline([
            ("scaler", RobustScaler()),
            ("clf",    LogisticRegression(
                C              = 10,
                max_iter       = 1000,
                class_weight   = "balanced",
                random_state   = 42,
                n_jobs         = -1
            ))
        ]),
    }

    print(f"  ✔ Built {len(models)} individual models:")
    for name in models:
        print(f"    · {name}")
    return models


# ─────────────────────────────────────────────────────────────────
# STEP 5 — CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────
def cross_validate_models(models: dict, X_train, y_train,
                           n_splits: int = 5) -> dict:
    print("\n" + "═" * 62)
    print(f"  STEP 5 · {n_splits}-FOLD STRATIFIED CROSS-VALIDATION")
    print("═" * 62)

    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    print(f"  {'Model':<22} {'Accuracy':>10}  {'± Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'─'*22} {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")

    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv, scoring="accuracy", n_jobs=-1)
        results[name] = scores
        status = "★ BEST" if scores.mean() == max(
            cross_val_score(m, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
            for m in models.values()
        ) else ""
        print(f"  {name:<22} {scores.mean():>10.4f}  {scores.std():>8.4f}"
              f"  {scores.min():>8.4f}  {scores.max():>8.4f}  {status}")

    return results


# ─────────────────────────────────────────────────────────────────
# STEP 6 — BUILD VOTING ENSEMBLE
# ─────────────────────────────────────────────────────────────────
def build_ensemble(models: dict, X_train, y_train,
                   cv_results: dict) -> Pipeline:
    print("\n" + "═" * 62)
    print("  STEP 6 · VOTING ENSEMBLE")
    print("═" * 62)

    # Select top-4 models by CV mean
    ranked = sorted(cv_results.items(), key=lambda x: x[1].mean(), reverse=True)
    top4   = ranked[:4]

    print(f"  Top-4 models selected for ensemble:")
    for name, scores in top4:
        print(f"    · {name:<22} CV={scores.mean():.4f}")

    estimators = [(name, models[name].named_steps["clf"])
                  for name, _ in top4]

    voting_clf = VotingClassifier(estimators=estimators, voting="soft")
    ensemble   = Pipeline([
        ("scaler", RobustScaler()),
        ("clf",    voting_clf)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ens_scores = cross_val_score(ensemble, X_train, y_train,
                                 cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"\n  Ensemble CV Accuracy : {ens_scores.mean():.4f} ± {ens_scores.std():.4f}")
    return ensemble, ens_scores


# ─────────────────────────────────────────────────────────────────
# STEP 7 — SELECT BEST & TRAIN FINAL MODEL
# ─────────────────────────────────────────────────────────────────
def train_final_model(models: dict, ensemble: Pipeline,
                      cv_results: dict, ensemble_scores,
                      X_train, y_train) -> tuple:
    print("\n" + "═" * 62)
    print("  STEP 7 · TRAINING FINAL MODEL")
    print("═" * 62)

    best_individual_name  = max(cv_results, key=lambda k: cv_results[k].mean())
    best_individual_score = cv_results[best_individual_name].mean()
    ens_score             = ensemble_scores.mean()

    print(f"  Best individual : {best_individual_name} (CV={best_individual_score:.4f})")
    print(f"  Ensemble        : Voting Ensemble     (CV={ens_score:.4f})")

    if ens_score >= best_individual_score:
        final_model = ensemble
        final_name  = "Voting Ensemble"
    else:
        final_model = models[best_individual_name]
        final_name  = best_individual_name

    print(f"\n  ★ Selected : {final_name}")
    print(f"  Training on full training set...")
    final_model.fit(X_train, y_train)
    print(f"  ✔ Training complete!")
    return final_model, final_name


# ─────────────────────────────────────────────────────────────────
# STEP 8 — EVALUATE ON TEST SET
# ─────────────────────────────────────────────────────────────────
def evaluate(final_model, final_name, X_test, y_test, le) -> dict:
    print("\n" + "═" * 62)
    print("  STEP 8 · TEST SET EVALUATION")
    print("═" * 62)

    y_pred  = final_model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    f1_mac  = f1_score(y_test, y_pred, average="macro")
    f1_wei  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  Model        : {final_name}")
    print(f"  Test Accuracy: {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Macro F1     : {f1_mac:.4f}")
    print(f"  Weighted F1  : {f1_wei:.4f}")
    print(f"\n  Per-Class Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_,
                                 zero_division=0))

    return {
        "y_pred": y_pred,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wei,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 9 — VISUALIZE RESULTS
# ─────────────────────────────────────────────────────────────────
def visualize(cv_results: dict, ensemble_scores, eval_results: dict,
              final_model, X_train, y_train, X_test, y_test,
              le, feature_names: list, final_name: str):
    print("\n" + "═" * 62)
    print("  STEP 9 · GENERATING VISUALIZATIONS")
    print("═" * 62)

    palette    = ["#FF6B35", "#9B59B6", "#2ECC71", "#3498DB", "#E74C3C"]
    y_pred     = eval_results["y_pred"]
    class_names = le.classes_

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0F1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    title_kw   = dict(color="white", fontsize=13, fontweight="bold", pad=12)
    label_kw   = dict(color="#AAAAAA", fontsize=10)
    tick_color = "#888888"

    def style_ax(ax, title):
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors=tick_color, labelsize=9)
        ax.spines[:].set_color("#333344")
        ax.set_title(title, **title_kw)

    # ── (A) Confusion Matrix ─────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    style_ax(ax0, "Confusion Matrix")
    cm  = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="#0F1117",
                ax=ax0, cbar_kws={"shrink": 0.8})
    ax0.set_xlabel("Predicted", **label_kw)
    ax0.set_ylabel("Actual",    **label_kw)
    ax0.tick_params(axis="x", rotation=30)

    # ── (B) CV Model Comparison ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    style_ax(ax1, "Model Accuracy Comparison (5-Fold CV)")

    all_names  = list(cv_results.keys()) + ["Voting Ensemble"]
    all_means  = [cv_results[k].mean() for k in cv_results] + [ensemble_scores.mean()]
    all_stds   = [cv_results[k].std()  for k in cv_results] + [ensemble_scores.std()]
    bar_colors = ["#3498DB"] * len(cv_results) + ["#F39C12"]

    bars = ax1.barh(all_names, all_means, xerr=all_stds,
                    color=bar_colors, edgecolor="#0F1117",
                    linewidth=0.5, capsize=4, height=0.6)
    ax1.set_xlim(0.75, 1.02)
    ax1.set_xlabel("Accuracy", **label_kw)
    for bar, val, std in zip(bars, all_means, all_stds):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", ha="left",
                 color="white", fontsize=8.5, fontweight="bold")
    ax1.axvline(x=max(all_means), color="#F39C12",
                linestyle="--", linewidth=1.0, alpha=0.7)

    # ── (C) Feature Importance ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, "Feature Importance (Random Forest)")

    rf_pipe = Pipeline([("scaler", RobustScaler()),
                         ("clf", RandomForestClassifier(
                             n_estimators=300, random_state=42, n_jobs=-1))])
    rf_pipe.fit(X_train, y_train)
    importances = rf_pipe.named_steps["clf"].feature_importances_
    idx         = np.argsort(importances)[::-1][:12]    # top-12

    colors_feat = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(idx)))
    ax2.barh([feature_names[i] for i in idx][::-1],
             importances[idx][::-1],
             color=colors_feat, edgecolor="#0F1117", linewidth=0.4)
    ax2.set_xlabel("Importance", **label_kw)

    # ── (D) Per-Class F1 Bar ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Per-Class F1 Score")

    from sklearn.metrics import f1_score
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    ax3.bar(class_names, per_class_f1, color=palette,
            edgecolor="#0F1117", linewidth=0.5)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("F1 Score", **label_kw)
    ax3.axhline(y=1.0, color="white", linestyle="--", linewidth=0.8, alpha=0.3)
    for i, v in enumerate(per_class_f1):
        ax3.text(i, v + 0.02, f"{v:.3f}", ha="center",
                 color="white", fontsize=9, fontweight="bold")
    ax3.tick_params(axis="x", rotation=15)

    # ── (E) CV Score Distribution (Box Plot) ────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "CV Score Distribution per Model")

    all_cv_scores = [cv_results[k] for k in cv_results] + [ensemble_scores]
    bp = ax4.boxplot(all_cv_scores, labels=all_names,
                     patch_artist=True, notch=False,
                     medianprops=dict(color="yellow", linewidth=2),
                     whiskerprops=dict(color="#AAAAAA"),
                     capprops=dict(color="#AAAAAA"),
                     flierprops=dict(markerfacecolor="#E74C3C",
                                     marker="o", markersize=5))
    for patch, c in zip(bp["boxes"], bar_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax4.set_ylabel("CV Accuracy", **label_kw)
    ax4.tick_params(axis="x", rotation=30, labelsize=7.5)
    ax4.set_ylim(0.75, 1.05)

    # ── (F) Summary Stats Panel ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1A1D27")
    ax5.spines[:].set_color("#333344")
    ax5.axis("off")

    stats_text = (
        f"FINAL RESULTS SUMMARY\n"
        f"{'─'*30}\n\n"
        f"Model        :  {final_name}\n\n"
        f"Test Accuracy:  {eval_results['accuracy']*100:.2f}%\n\n"
        f"Macro F1     :  {eval_results['f1_macro']:.4f}\n\n"
        f"Weighted F1  :  {eval_results['f1_weighted']:.4f}\n\n"
        f"{'─'*30}\n\n"
        f"Classes      :  {len(class_names)}\n\n"
        f"Dataset Size :  440 samples\n\n"
        f"Train / Test :  352 / 88\n\n"
        f"CV Folds     :  5 (Stratified)\n\n"
        f"{'─'*30}\n\n"
        f"Best CV Acc  :  {max(v.mean() for v in cv_results.values()):.4f}\n\n"
        f"Ensemble CV  :  {ensemble_scores.mean():.4f}"
    )
    ax5.text(0.08, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=10.5, verticalalignment="top",
             color="white", fontfamily="monospace",
             linespacing=1.3,
             bbox=dict(boxstyle="round,pad=0.5",
                        facecolor="#0D2137",
                        edgecolor="#3498DB",
                        linewidth=1.5))

    fig.text(0.5, 0.985,
             "SMART IRON — FABRIC CLASSIFIER — TRAINING REPORT",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color="white")

    plt.savefig("smart_iron_training_report.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1117")
    print("  ✔ Saved: smart_iron_training_report.png")


# ─────────────────────────────────────────────────────────────────
# STEP 10 — SAVE MODEL
# ─────────────────────────────────────────────────────────────────
def save_model(final_model, le, feature_names: list, final_name: str):
    print("\n" + "═" * 62)
    print("  STEP 10 · SAVING MODEL")
    print("═" * 62)

    bundle = {
        "model":         final_model,
        "label_encoder": le,
        "feature_names": feature_names,
        "model_name":    final_name,
    }
    path = "smart_iron_model.pkl"
    joblib.dump(bundle, path)
    size = os.path.getsize(path) / 1024
    print(f"  ✔ Model saved : {path}  ({size:.1f} KB)")
    return path


# ─────────────────────────────────────────────────────────────────
# STEP 11 — PREDICT FUNCTION
# ─────────────────────────────────────────────────────────────────
def load_and_predict(model_path: str,
                     temperature_c: float,
                     motion_variation: float,
                     static_time_s: int) -> dict:
    """
    Load saved model and predict fabric type from raw sensor inputs.

    Parameters
    ----------
    model_path      : Path to smart_iron_model.pkl
    temperature_c   : Iron plate temperature in Celsius
    motion_variation: IMU-based motion variation (float, 0–0.05)
    static_time_s   : Seconds the iron was kept stationary

    Returns
    -------
    dict with keys: fabric, confidence, all_probabilities
    """
    bundle   = joblib.load(model_path)
    model    = bundle["model"]
    le       = bundle["label_encoder"]
    eps      = 1e-9

    T, M, S = temperature_c, motion_variation, static_time_s

    features = {
        "Temperature_C":       T,
        "Motion_Variation":    M,
        "Static_Time_s":       S,
        "Temp_Motion_Ratio":   T / (M + eps),
        "Temp_Static_Product": T * S,
        "Motion_Static_Ratio": M / (S + eps),
        "Temp_Motion_Product": T * M,
        "Motion_log":          np.log1p(M),
        "Static_log":          np.log1p(S),
        "Temp_squared":        T ** 2,
        "Motion_squared":      M ** 2,
        "Heat_Exposure_Index": T * S / (M + eps),
    }

    X_input  = np.array([list(features.values())])
    pred_idx = model.predict(X_input)[0]
    proba    = model.predict_proba(X_input)[0]
    label    = le.inverse_transform([pred_idx])[0]

    result = {
        "fabric":             label,
        "confidence":         round(float(proba.max()) * 100, 2),
        "all_probabilities":  dict(zip(le.classes_, np.round(proba * 100, 2))),
    }
    return result


# ─────────────────────────────────────────────────────────────────
# MAIN — RUN FULL PIPELINE
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "╔" + "═"*60 + "╗")
    print("║" + "  SMART IRON · FABRIC CLASSIFIER · TRAINING PIPELINE".center(60) + "║")
    print("╚" + "═"*60 + "╝")

    # ── 1. Load ──────────────────────────────────────────────────
    df = load_data("smart_iron_dataset_corrected.xlsx")

    # ── 2. Feature Engineering ───────────────────────────────────
    df_eng = engineer_features(df)

    # ── 3. Encode & Split ────────────────────────────────────────
    X_train, X_test, y_train, y_test, le = encode_and_split(df_eng)
    feature_names = df_eng.drop(columns=["Fabric_Type"]).columns.tolist()

    # ── 4. Build Models ──────────────────────────────────────────
    models = build_models()

    # ── 5. Cross-Validate ────────────────────────────────────────
    cv_results = cross_validate_models(models, X_train, y_train)

    # ── 6. Build Ensemble ────────────────────────────────────────
    ensemble, ens_scores = build_ensemble(models, X_train, y_train, cv_results)

    # ── 7. Select & Train Final Model ────────────────────────────
    final_model, final_name = train_final_model(
        models, ensemble, cv_results, ens_scores, X_train, y_train
    )

    # ── 8. Evaluate on Test Set ──────────────────────────────────
    eval_results = evaluate(final_model, final_name, X_test, y_test, le)

    # ── 9. Visualize ─────────────────────────────────────────────
    visualize(cv_results, ens_scores, eval_results,
              final_model, X_train, y_train, X_test, y_test,
              le, feature_names, final_name)

    # ── 10. Save Model ───────────────────────────────────────────
    model_path = save_model(final_model, le, feature_names, final_name)

    # ── 11. Demo Predictions ─────────────────────────────────────
    print("\n" + "═" * 62)
    print("  STEP 11 · SAMPLE PREDICTIONS (from saved model)")
    print("═" * 62)

    test_cases = [
        (192.0, 0.042, 5,  "Cotton (expected)"),
        (128.0, 0.009, 3,  "Silk (expected)"),
        (136.0, 0.031, 6,  "Wool (expected)"),
        (130.0, 0.019, 7,  "Polyester (expected)"),
        (235.0, 0.005, 30, "Anomaly — Too Hot + Static (expected)"),
        (75.0,  0.040, 25, "Anomaly — Too Cold (expected)"),
    ]

    print(f"\n  {'Input':^42} {'Predicted':<14} {'Conf':>7}  {'Expected'}")
    print(f"  {'─'*42} {'─'*14} {'─'*7}  {'─'*25}")

    for temp, motion, static, expected in test_cases:
        result = load_and_predict(model_path, temp, motion, static)
        match  = "✔" if result["fabric"].lower() in expected.lower() else "✗"
        print(f"  T={temp:>5.1f}°C M={motion:.3f} S={static:>2}s  "
              f"→  {result['fabric']:<14} {result['confidence']:>6.1f}%  "
              f"{match} {expected}")

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "╔" + "═"*60 + "╗")
    print("║" + "  TRAINING COMPLETE!".center(60) + "║")
    print(f"║{'':4}Final Model    : {final_name:<42}║")
    print(f"║{'':4}Test Accuracy  : {eval_results['accuracy']*100:.2f}%{'':<42}║")
    print(f"║{'':4}Weighted F1    : {eval_results['f1_weighted']:.4f}{'':<42}║")
    print(f"║{'':4}Model File     : smart_iron_model.pkl{'':<23}║")
    print(f"║{'':4}Report Plot    : smart_iron_training_report.png{'':<13}║")
    print("╚" + "═"*60 + "╝\n")


if __name__ == "__main__":
    main()
