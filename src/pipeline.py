"""
CreditGuard End-to-End Pipeline:
1. Stratified 80/20 data split and preprocessing.
2. Loan Amount Regression (Normal Equation & Gradient Descent) with R^2, RMSE, residual analysis.
3. PCA Scree plot and 2D default-risk projection.
4. Classification benchmark (DecisionTree, AdaBoost, SVM) on Raw vs PCA features,
   highlighting Recall on the Bad Credit class.
5. Customer Segmentation (EM-GMM vs Divisive Graph Clustering) and financial profiling.
6. Export of visual artifacts and summary tables to report/.
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for generating figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier as SklearnDTC
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.svm import LinearSVC as SklearnLinearSVC
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

import sys
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.data_loader import get_preprocessed_credit_data
from src.linear_regression import MultipleLinearRegression, SimpleLinearRegression
from src.pca import PCA
from src.decision_tree import DecisionTreeClassifier
from src.adaboost import AdaBoostClassifier
from src.svm import LinearSVM
from src.gmm import GaussianMixtureModel
from src.graph_clustering import DivisiveGraphClustering

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "report")


def run_pipeline() -> Dict[str, Any]:
    os.makedirs(REPORT_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    print("=" * 70)
    print("STEP 1: Loading & Preprocessing German Credit Dataset...")
    print("=" * 70)
    data = get_preprocessed_credit_data(test_size=0.2, random_state=42)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    X_reg_train, X_reg_test = data["X_reg_train"], data["X_reg_test"]
    y_reg_train, y_reg_test = data["y_reg_train"], data["y_reg_test"]
    df_train_raw, df_test_raw = data["df_train_raw"], data["df_test_raw"]

    print(f"Train samples: {len(X_train)} (Bad credit: {np.sum(y_train)} [{np.mean(y_train):.1%}])")
    print(f"Test samples:  {len(X_test)} (Bad credit: {np.sum(y_test)} [{np.mean(y_test):.1%}])")
    print(f"Classification features: {X_train.shape[1]}")
    print(f"Regression features:     {X_reg_train.shape[1]}")

    # =========================================================================
    # STEP 2: REGRESSION (Predict Loan Amount)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Multiple Linear Regression (Predicting Loan Amount)...")
    print("=" * 70)

    # 1. Normal Equation
    reg_normal = MultipleLinearRegression(method="normal", l2_reg=1e-5).fit(
        X_reg_train, y_reg_train
    )
    r2_train_norm = reg_normal.score(X_reg_train, y_reg_train)
    r2_test_norm = reg_normal.score(X_reg_test, y_reg_test)
    rmse_test_norm = reg_normal.rmse(X_reg_test, y_reg_test)
    y_pred_reg_test = reg_normal.predict(X_reg_test)
    residuals = y_reg_test - y_pred_reg_test

    # 2. Sklearn benchmark
    sk_reg = SklearnLinearRegression().fit(X_reg_train, y_reg_train)
    sk_r2_test = sk_reg.score(X_reg_test, y_reg_test)

    # 3. Simple Linear Regression (Duration vs Loan Amount)
    idx_duration = data["reg_feature_names"].index("duration")
    simple_reg = SimpleLinearRegression().fit(X_reg_train[:, idx_duration], y_reg_train)
    simple_r2 = simple_reg.score(X_reg_test[:, idx_duration], y_reg_test)

    print(f"Multiple Regression (Normal Equation) Test R^2: {r2_test_norm:.4f} (Sklearn: {sk_r2_test:.4f})")
    print(f"Multiple Regression Test RMSE:                  ${rmse_test_norm:,.2f}")
    print(f"Simple Regression (Duration only) Test R^2:     {simple_r2:.4f}")

    # Plot Residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred_reg_test, residuals, alpha=0.6, color="#4F46E5", edgecolors="none")
    axes[0].axhline(0, color="#EF4444", linestyle="--", linewidth=1.5)
    axes[0].set_title("Residuals vs. Predicted Loan Amount", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Loan Amount ($)")
    axes[0].set_ylabel("Residual ($)")

    sns.histplot(residuals, kde=True, ax=axes[1], color="#06B6D4", bins=25)
    axes[1].axvline(0, color="#EF4444", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residual Distribution (Prediction Errors)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Residual ($)")

    plt.tight_layout()
    res_plot_path = os.path.join(REPORT_DIR, "regression_residuals.png")
    fig.savefig(res_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved residual plot to: {res_plot_path}")

    # =========================================================================
    # STEP 3: PCA (Scree Plot & 2D Projection)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: PCA Dimensionality Reduction & Variance Analysis...")
    print("=" * 70)

    pca_full = PCA(n_components=15).fit(X_train)
    exp_var = pca_full.explained_variance_ratio_
    cum_var = pca_full.cumulative_explained_variance_ratio_

    print(f"Top 2 PCs explain: {cum_var[1]:.1%} of variance")
    print(f"Top 5 PCs explain: {cum_var[4]:.1%} of variance")
    print(f"Top 10 PCs explain: {cum_var[9]:.1%} of variance")

    # Scree Plot
    fig, ax1 = plt.subplots(figsize=(9, 5))
    comp_range = np.arange(1, len(exp_var) + 1)
    bars = ax1.bar(comp_range, exp_var * 100, color="#6366F1", alpha=0.8, label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component", fontweight="bold")
    ax1.set_ylabel("Explained Variance Ratio (%)", color="#6366F1", fontweight="bold")
    ax1.set_xticks(comp_range)

    ax2 = ax1.twinx()
    ax2.plot(comp_range, cum_var * 100, color="#EC4899", marker="o", linewidth=2.2, label="Cumulative (%)")
    ax2.set_ylabel("Cumulative Variance (%)", color="#EC4899", fontweight="bold")
    ax2.grid(False)

    plt.title("PCA Scree Plot & Cumulative Explained Variance", fontsize=13, fontweight="bold", pad=12)
    scree_plot_path = os.path.join(REPORT_DIR, "pca_scree_plot.png")
    fig.savefig(scree_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved PCA scree plot to: {scree_plot_path}")

    # 2D Scatter colored by Default Label
    pca_2d = PCA(n_components=2).fit(X_train)
    X_train_pca2 = pca_2d.transform(X_train)
    X_test_pca2 = pca_2d.transform(X_test)

    fig, ax = plt.subplots(figsize=(9, 6))
    good_mask = y_train == 0
    bad_mask = y_train == 1
    ax.scatter(
        X_train_pca2[good_mask, 0],
        X_train_pca2[good_mask, 1],
        c="#10B981",
        label="Good Credit (0)",
        alpha=0.6,
        edgecolors="none",
        s=35,
    )
    ax.scatter(
        X_train_pca2[bad_mask, 0],
        X_train_pca2[bad_mask, 1],
        c="#EF4444",
        label="Bad Credit / Default (1)",
        alpha=0.75,
        edgecolors="k",
        linewidths=0.5,
        s=45,
    )
    ax.set_title("Customer Risk Distribution in 2D Principal Space", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(f"PC1 ({exp_var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({exp_var[1]:.1%} variance)")
    ax.legend(loc="upper right", frameon=True)
    scatter_plot_path = os.path.join(REPORT_DIR, "pca_2d_scatter.png")
    fig.savefig(scatter_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved 2D PCA scatter to: {scatter_plot_path}")

    # Transform to 10 PCs for model benchmarking
    pca_10 = PCA(n_components=10).fit(X_train)
    X_train_pca10 = pca_10.transform(X_train)
    X_test_pca10 = pca_10.transform(X_test)

    # =========================================================================
    # STEP 4: CLASSIFICATION BENCHMARK (Raw vs PCA)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Training & Benchmarking Classifiers (Raw vs PCA Features)...")
    print("=" * 70)

    models_config = [
        (
            "Decision Tree (Scratch)",
            DecisionTreeClassifier(max_depth=5, min_samples_split=10),
            DecisionTreeClassifier(max_depth=5, min_samples_split=10),
        ),
        (
            "AdaBoost (Scratch)",
            AdaBoostClassifier(n_estimators=35, learning_rate=0.7),
            AdaBoostClassifier(n_estimators=35, learning_rate=0.7),
        ),
        (
            "Linear SVM (Scratch)",
            LinearSVM(C=1.0, learning_rate=0.005, epochs=300, random_state=42),
            LinearSVM(C=1.0, learning_rate=0.005, epochs=300, random_state=42),
        ),
        (
            "Decision Tree (Sklearn)",
            SklearnDTC(max_depth=5, min_samples_split=10, random_state=42),
            SklearnDTC(max_depth=5, min_samples_split=10, random_state=42),
        ),
        (
            "AdaBoost (Sklearn)",
            SklearnAdaBoost(n_estimators=35, learning_rate=0.7, random_state=42),
            SklearnAdaBoost(n_estimators=35, learning_rate=0.7, random_state=42),
        ),
        (
            "Linear SVM (Sklearn)",
            SklearnLinearSVC(C=1.0, random_state=42, dual="auto"),
            SklearnLinearSVC(C=1.0, random_state=42, dual="auto"),
        ),
    ]

    benchmark_records = []
    roc_curves_data = {}

    for name, model_raw, model_pca in models_config:
        # 1. Fit on Raw features
        model_raw.fit(X_train, y_train)
        preds_raw = model_raw.predict(X_test)
        if hasattr(model_raw, "predict_proba"):
            probs_raw = model_raw.predict_proba(X_test)[:, 1]
        elif hasattr(model_raw, "decision_function"):
            probs_raw = model_raw.decision_function(X_test)
        else:
            probs_raw = preds_raw

        acc_raw = accuracy_score(y_test, preds_raw)
        prec_raw = precision_score(y_test, preds_raw, zero_division=0)
        rec_raw = recall_score(y_test, preds_raw)
        f1_raw = f1_score(y_test, preds_raw, zero_division=0)
        try:
            auc_raw = roc_auc_score(y_test, probs_raw)
        except Exception:
            auc_raw = 0.5

        benchmark_records.append({
            "Model": name,
            "Feature Set": "Raw (61 cols)",
            "Accuracy": acc_raw,
            "Precision (Bad)": prec_raw,
            "Recall (Bad) [KEY]": rec_raw,
            "F1-Score": f1_raw,
            "ROC-AUC": auc_raw,
        })

        # 2. Fit on PCA features (top 10 components)
        model_pca.fit(X_train_pca10, y_train)
        preds_pca = model_pca.predict(X_test_pca10)
        if hasattr(model_pca, "predict_proba"):
            probs_pca = model_pca.predict_proba(X_test_pca10)[:, 1]
        elif hasattr(model_pca, "decision_function"):
            probs_pca = model_pca.decision_function(X_test_pca10)
        else:
            probs_pca = preds_pca

        acc_pca = accuracy_score(y_test, preds_pca)
        prec_pca = precision_score(y_test, preds_pca, zero_division=0)
        rec_pca = recall_score(y_test, preds_pca)
        f1_pca = f1_score(y_test, preds_pca, zero_division=0)
        try:
            auc_pca = roc_auc_score(y_test, probs_pca)
        except Exception:
            auc_pca = 0.5

        benchmark_records.append({
            "Model": name,
            "Feature Set": "PCA (10 comps)",
            "Accuracy": acc_pca,
            "Precision (Bad)": prec_pca,
            "Recall (Bad) [KEY]": rec_pca,
            "F1-Score": f1_pca,
            "ROC-AUC": auc_pca,
        })

        if "Scratch" in name:
            roc_curves_data[name] = (y_test, probs_raw)

    df_benchmark = pd.DataFrame(benchmark_records)
    csv_benchmark_path = os.path.join(REPORT_DIR, "classification_benchmark.csv")
    df_benchmark.to_csv(csv_benchmark_path, index=False)

    print("\n--- MODEL CLASSIFICATION BENCHMARK RESULTS ---")
    print(df_benchmark.to_string(index=False))

    # ROC Curves Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_t, p_t) in roc_curves_data.items():
        fpr, tpr, _ = roc_curve(y_t, p_t)
        auc = roc_auc_score(y_t, p_t)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2.0)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Baseline (AUC = 0.50)")
    ax.set_title("ROC Curves for Credit Default Detection (Scratch Models)", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall on Bad Credit)")
    ax.legend(loc="lower right")
    roc_plot_path = os.path.join(REPORT_DIR, "roc_curves.png")
    fig.savefig(roc_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved ROC curves plot to: {roc_plot_path}")

    # =========================================================================
    # STEP 5: CUSTOMER SEGMENTATION & CLUSTER PROFILING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Customer Segmentation (EM-GMM vs Divisive Graph Clustering)...")
    print("=" * 70)

    # Use PCA 2D/3D representation of full training cohort for clustering
    n_clusters = 3
    gmm = GaussianMixtureModel(n_components=n_clusters, max_iter=60, random_state=42)
    gmm.fit(X_train_pca2)
    gmm_clusters = gmm.predict(X_train_pca2)

    # Divisive Graph Clustering on PCA representation
    # Use representative subsample (e.g. 200 points) or full to ensure fast BFS betweenness
    sub_idx = np.arange(len(X_train))
    dgc = DivisiveGraphClustering(n_clusters=n_clusters, n_neighbors=8, criterion="n_clusters")
    dgc.fit(X_train_pca2[:300])  # 300 points for crisp graph partitioning
    dgc_clusters = dgc.predict(X_train_pca2)

    # Profiling dataframe on training cohort
    df_profile = df_train_raw.copy()
    df_profile["GMM_Cluster"] = gmm_clusters
    df_profile["Graph_Cluster"] = dgc_clusters
    df_profile["is_bad_credit"] = y_train

    def compute_cluster_profiles(cluster_col: str, method_name: str) -> pd.DataFrame:
        records = []
        for c in range(n_clusters):
            sub = df_profile[df_profile[cluster_col] == c]
            records.append({
                "Method": method_name,
                "Segment": f"Cluster {c}",
                "Count": len(sub),
                "Pct Cohort": f"{len(sub) / len(df_profile):.1%}",
                "Mean Loan Amount ($)": f"${sub['credit_amount'].mean():,.0f}",
                "Mean Duration (mo)": f"{sub['duration'].mean():.1f}",
                "Mean Age (yr)": f"{sub['age'].mean():.1f}",
                "Installment Rate": f"{sub['installment_commitment'].mean():.2f}",
                "Default Rate (%)": f"{sub['is_bad_credit'].mean():.1%}",
            })
        return pd.DataFrame(records)

    gmm_profiles = compute_cluster_profiles("GMM_Cluster", "EM-GMM")
    graph_profiles = compute_cluster_profiles("Graph_Cluster", "Divisive Graph")
    df_cluster_profiles = pd.concat([gmm_profiles, graph_profiles], ignore_index=True)

    print("\n--- CUSTOMER SEGMENT PROFILES ---")
    print(df_cluster_profiles.to_string(index=False))

    # Cluster Visual Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    palette = ["#3B82F6", "#10B981", "#F59E0B"]

    # GMM plot
    for c in range(n_clusters):
        c_mask = gmm_clusters == c
        axes[0].scatter(
            X_train_pca2[c_mask, 0],
            X_train_pca2[c_mask, 1],
            c=palette[c],
            label=f"Cluster {c} (Def: {df_profile[df_profile['GMM_Cluster']==c]['is_bad_credit'].mean():.1%})",
            alpha=0.65,
            edgecolors="none",
            s=40,
        )
    # Plot GMM means
    axes[0].scatter(
        gmm.means_[:, 0],
        gmm.means_[:, 1],
        c="red",
        marker="X",
        s=120,
        edgecolors="black",
        label="Cluster Means",
    )
    axes[0].set_title("EM-Gaussian Mixture Model Segments", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(loc="upper right", fontsize=9)

    # Divisive Graph plot
    for c in range(n_clusters):
        c_mask = dgc_clusters == c
        axes[1].scatter(
            X_train_pca2[c_mask, 0],
            X_train_pca2[c_mask, 1],
            c=palette[c],
            label=f"Cluster {c} (Def: {df_profile[df_profile['Graph_Cluster']==c]['is_bad_credit'].mean():.1%})",
            alpha=0.65,
            edgecolors="none",
            s=40,
        )
    axes[1].set_title("Divisive Graph Clustering (Girvan-Newman)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    cluster_plot_path = os.path.join(REPORT_DIR, "clusters_comparison.png")
    fig.savefig(cluster_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved cluster comparison plot to: {cluster_plot_path}")

    return {
        "df_benchmark": df_benchmark,
        "df_cluster_profiles": df_cluster_profiles,
        "regression_metrics": {
            "r2_test": r2_test_norm,
            "rmse_test": rmse_test_norm,
            "simple_r2": simple_r2,
        },
        "pca_metrics": {
            "cum_var_2": cum_var[1],
            "cum_var_5": cum_var[4],
            "cum_var_10": cum_var[9],
        },
    }


if __name__ == "__main__":
    results = run_pipeline()
    print("\nPipeline executed successfully!")
