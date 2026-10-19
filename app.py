"""
CreditGuard: Interactive Credit-Risk Assessment and Customer Segmentation Application.
Built with Streamlit and powered by algorithms implemented from scratch in NumPy.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.data_loader import (
    get_preprocessed_credit_data,
    CATEGORICAL_COLS,
    NUMERICAL_COLS_ALL,
)
from src.linear_regression import MultipleLinearRegression, SimpleLinearRegression
from src.pca import PCA
from src.decision_tree import DecisionTreeClassifier
from src.adaboost import AdaBoostClassifier
from src.svm import LinearSVM
from src.gmm import GaussianMixtureModel
from src.graph_clustering import DivisiveGraphClustering

# Page Configuration
st.set_page_config(
    page_title="CreditGuard | Risk & Segmentation Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Financial Dashboard Theme)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .risk-badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training scratch NumPy models on German Credit data...")
def load_and_train_models():
    """Load dataset, fit preprocessor, and train all scratch algorithms."""
    data = get_preprocessed_credit_data(test_size=0.2, random_state=42)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_reg_train = data["X_reg_train"]
    y_reg_train = data["y_reg_train"]

    # 1. Multiple Linear Regression for Loan Amount
    reg_model = MultipleLinearRegression(method="normal", l2_reg=1e-5).fit(
        X_reg_train, y_reg_train
    )

    # 2. PCA
    pca_2d = PCA(n_components=2).fit(X_train)
    pca_10d = PCA(n_components=10).fit(X_train)
    X_pca2 = pca_2d.transform(X_train)

    # 3. Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10).fit(
        X_train, y_train
    )

    # 4. AdaBoost (Ensemble)
    ada_model = AdaBoostClassifier(n_estimators=35, learning_rate=0.7).fit(
        X_train, y_train
    )

    # 5. Linear SVM
    svm_model = LinearSVM(C=1.0, learning_rate=0.005, epochs=250, random_state=42).fit(
        X_train, y_train
    )

    # 6. EM Gaussian Mixture Model
    gmm_model = GaussianMixtureModel(n_components=3, max_iter=50, random_state=42).fit(
        X_pca2
    )

    # 7. Divisive Graph Clustering
    dgc_model = DivisiveGraphClustering(n_clusters=3, n_neighbors=8).fit(X_pca2[:300])

    return {
        "data": data,
        "reg_model": reg_model,
        "pca_2d": pca_2d,
        "pca_10d": pca_10d,
        "dt_model": dt_model,
        "ada_model": ada_model,
        "svm_model": svm_model,
        "gmm_model": gmm_model,
        "dgc_model": dgc_model,
        "X_pca2": X_pca2,
    }


models_bundle = load_and_train_models()
data_dict = models_bundle["data"]
preprocessor = data_dict["preprocessor"]
reg_preprocessor = data_dict["reg_preprocessor"]
raw_df = data_dict["raw_df"]

# Header
st.markdown('<div class="main-title">CreditGuard Lending Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Production Credit-Risk Underwriting and Customer Segmentation engine powered by NumPy first-principles algorithms.</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Applicant Risk Assessment",
        "📊 Model Benchmarks & Comparison",
        "🧭 PCA & Risk Space",
        "👥 Customer Segmentation",
        "🌳 Decision Tree Rules",
    ]
)

# =============================================================================
# TAB 1: APPLICANT RISK ASSESSMENT & LOAN ADVISORY
# =============================================================================
with tab1:
    st.subheader("New Loan Application Assessment")
    st.write(
        "Enter borrower characteristics below to evaluate real-time credit default risk, safe loan sizing, and persona segmentation."
    )

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 💼 Financial Baseline")
            duration = st.slider("Loan Duration (Months)", min_value=4, max_value=72, value=24, step=2)
            credit_amount = st.number_input(
                "Requested Credit Amount ($)", min_value=250, max_value=20000, value=3500, step=100
            )
            checking_status = st.selectbox(
                "Checking Account Status",
                options=sorted(list(raw_df["checking_status"].unique())),
                index=0,
                help="Current checking balance status in bank account.",
            )
            savings_status = st.selectbox(
                "Savings Account Status",
                options=sorted(list(raw_df["savings_status"].unique())),
                index=0,
                help="Reported savings / liquid bonds.",
            )
            installment_commitment = st.slider(
                "Installment Rate (% of disposable income)", min_value=1, max_value=4, value=3
            )

        with col2:
            st.markdown("##### 👤 Applicant Demographics")
            age = st.slider("Borrower Age (Years)", min_value=18, max_value=75, value=35)
            personal_status = st.selectbox(
                "Personal Status & Gender",
                options=sorted(list(raw_df["personal_status"].unique())),
                index=0,
            )
            employment = st.selectbox(
                "Employment Length",
                options=sorted(list(raw_df["employment"].unique())),
                index=2,
            )
            housing = st.selectbox(
                "Housing Type",
                options=sorted(list(raw_df["housing"].unique())),
                index=1,
            )
            job = st.selectbox(
                "Employment Category",
                options=sorted(list(raw_df["job"].unique())),
                index=2,
            )

        with col3:
            st.markdown("##### 📑 Credit History & Terms")
            credit_history = st.selectbox(
                "Credit History",
                options=sorted(list(raw_df["credit_history"].unique())),
                index=1,
            )
            purpose = st.selectbox(
                "Loan Purpose",
                options=sorted(list(raw_df["purpose"].unique())),
                index=1,
            )
            other_parties = st.selectbox(
                "Guarantor / Co-Applicant",
                options=sorted(list(raw_df["other_parties"].unique())),
                index=0,
            )
            existing_credits = st.slider("Existing Bank Credits", min_value=1, max_value=4, value=1)
            other_payment_plans = st.selectbox(
                "Other Installment Plans",
                options=sorted(list(raw_df["other_payment_plans"].unique())),
                index=1,
            )

        # Advanced/Secondary toggles in an expander
        with st.expander("Additional Applicant Attributes (Residence, Dependents, Foreign Worker)"):
            c_a, c_b, c_c, c_d = st.columns(4)
            with c_a:
                residence_since = st.slider("Years at Present Residence", 1, 4, 2)
            with c_b:
                num_dependents = st.selectbox("Number of Dependents", [1, 2], index=0)
            with c_c:
                own_telephone = st.selectbox("Registered Telephone", ["yes", "none"], index=0)
            with c_d:
                foreign_worker = st.selectbox("Foreign Worker Status", ["yes", "no"], index=0)
            property_magnitude = st.selectbox(
                "Property Magnitude",
                options=sorted(list(raw_df["property_magnitude"].unique())),
                index=0,
            )

        submit_btn = st.form_submit_button("⚡ Run Credit Underwriting Analysis", use_container_width=True)

    if submit_btn:
        # Build dictionary matching raw dataframe column structure
        applicant_dict = {
            "checking_status": checking_status,
            "duration": duration,
            "credit_history": credit_history,
            "purpose": purpose,
            "credit_amount": credit_amount,
            "savings_status": savings_status,
            "employment": employment,
            "installment_commitment": installment_commitment,
            "personal_status": personal_status,
            "other_parties": other_parties,
            "residence_since": residence_since,
            "property_magnitude": property_magnitude,
            "age": age,
            "other_payment_plans": other_payment_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "num_dependents": num_dependents,
            "own_telephone": own_telephone,
            "foreign_worker": foreign_worker,
        }

        # Transform applicant feature vector
        X_applicant = preprocessor.transform_single_applicant(applicant_dict)
        X_applicant_reg = reg_preprocessor.transform_single_applicant(applicant_dict)
        X_applicant_pca2 = models_bundle["pca_2d"].transform(X_applicant)

        # 1. Predictions from Scratch Models
        ada_probs = models_bundle["ada_model"].predict_proba(X_applicant)[0]
        dt_probs = models_bundle["dt_model"].predict_proba(X_applicant)[0]
        svm_probs = models_bundle["svm_model"].predict_proba(X_applicant)[0]

        # Ensemble default probability (weighted toward AdaBoost)
        p_default = float(0.6 * ada_probs[1] + 0.2 * dt_probs[1] + 0.2 * svm_probs[1])

        # 2. Regression baseline for suggested loan amount
        suggested_baseline = float(models_bundle["reg_model"].predict(X_applicant_reg)[0])
        # Safety haircut based on risk probability
        safe_loan_amount = max(500.0, suggested_baseline * (1.0 - 0.5 * p_default))

        # 3. Customer Segment assignment
        gmm_seg = int(models_bundle["gmm_model"].predict(X_applicant_pca2)[0])
        dgc_seg = int(models_bundle["dgc_model"].predict(X_applicant_pca2)[0])

        segment_names = {
            0: "Young Entry-Level / Moderate Risk",
            1: "Capital-Intensive / High Exposure",
            2: "Prime Conservative / Low Risk",
        }

        st.markdown("---")
        st.markdown("### 🎯 Underwriting Decision & Advisory")

        col_dec1, col_dec2, col_dec3, col_dec4 = st.columns(4)

        with col_dec1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Default Probability", f"{p_default:.1%}")
            if p_default < 0.28:
                st.markdown('<span class="risk-badge-low">LOW RISK</span>', unsafe_allow_html=True)
            elif p_default < 0.45:
                st.markdown('<span class="risk-badge-med">MODERATE RISK</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="risk-badge-high">HIGH RISK</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_dec2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Requested Loan", f"${credit_amount:,.0f}")
            status_text = "APPROVED" if p_default < 0.35 else ("REVIEW" if p_default < 0.50 else "DECLINE")
            st.write(f"Recommendation: **{status_text}**")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_dec3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Suggested Safe Loan Cap", f"${safe_loan_amount:,.0f}")
            diff = safe_loan_amount - credit_amount
            st.caption(f"{'+' if diff >= 0 else ''}${diff:,.0f} vs requested amount")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_dec4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Assigned Customer Segment", f"Segment {gmm_seg}")
            st.caption(segment_names.get(gmm_seg, f"Cluster {gmm_seg}"))
            st.markdown("</div>", unsafe_allow_html=True)

        # Detailed Diagnostic Breakdown
        st.markdown("#### Model Breakdown for Applicant")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.info(f"**AdaBoost Default Risk:** {ada_probs[1]:.1%}")
        col_m2.info(f"**Decision Tree Default Risk:** {dt_probs[1]:.1%}")
        col_m3.info(f"**Linear SVM Default Risk:** {svm_probs[1]:.1%}")

# =============================================================================
# TAB 2: MODEL BENCHMARKS & COMPARISON
# =============================================================================
with tab2:
    st.subheader("Model Performance Benchmark: Scratch NumPy vs. Scikit-Learn")
    st.write(
        "Direct performance comparison on the 20% holdout test set (200 credit applicants: 140 Good, 60 Bad). "
        "Notice that **Recall on Bad Credit** is the primary lending KPI because an unhedged default incurs severe capital loss."
    )

    csv_path = os.path.join(WORKSPACE_ROOT, "report", "classification_benchmark.csv")
    if os.path.exists(csv_path):
        df_bench = pd.read_csv(csv_path)

        # Format dataframe for display
        df_display = df_bench.copy()
        for col in ["Accuracy", "Precision (Bad)", "Recall (Bad) [KEY]", "F1-Score", "ROC-AUC"]:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")

        st.dataframe(df_display, use_container_width=True)

        # Visual Comparison Bar Chart
        st.markdown("#### Key Metric Comparison: Recall on Bad Credit & ROC-AUC")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            raw_subset = df_bench[df_bench["Feature Set"] == "Raw (61 cols)"]
            sns.barplot(data=raw_subset, x="Recall (Bad) [KEY]", y="Model", ax=ax, palette="mako")
            ax.set_title("Recall on Bad Credit (Raw Features)", fontweight="bold")
            ax.set_xlim(0, 1.0)
            st.pyplot(fig)

        with col_c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=raw_subset, x="ROC-AUC", y="Model", ax=ax, palette="viridis")
            ax.set_title("ROC-AUC Discrimination Power", fontweight="bold")
            ax.set_xlim(0.5, 1.0)
            st.pyplot(fig)

        # Display ROC Curve image
        roc_img = os.path.join(WORKSPACE_ROOT, "report", "roc_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Holdout ROC Curves for Scratch NumPy Classifiers", width=700)

        # Financial Risk Tradeoff Note
        st.info(
            "💡 **Lending Economics Note**: At default rate = 30%, catching a bad loan preserves $2,500 - $6,000 in capital, "
            "whereas declining a good loan merely forfeits ~10% interest ($250 - $600). "
            "Underwriters can lower the probability decision threshold from 0.50 to 0.35 to achieve >75% Bad-Credit Recall."
        )

# =============================================================================
# TAB 3: PCA & RISK SPACE
# =============================================================================
with tab3:
    st.subheader("Principal Component Analysis (PCA) & Variance Decomposition")
    st.write(
        "Dimensionality reduction implemented via sample covariance matrix and eigendecomposition (NumPy only)."
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        scree_img = os.path.join(WORKSPACE_ROOT, "report", "pca_scree_plot.png")
        if os.path.exists(scree_img):
            st.image(scree_img, caption="PCA Scree Plot & Cumulative Explained Variance", use_container_width=True)

    with col_p2:
        scatter_img = os.path.join(WORKSPACE_ROOT, "report", "pca_2d_scatter.png")
        if os.path.exists(scatter_img):
            st.image(scatter_img, caption="2D Customer Distribution Colored by Default Label", use_container_width=True)

    st.markdown("#### Regression Residual Analysis (Loan Amount Prediction)")
    res_img = os.path.join(WORKSPACE_ROOT, "report", "regression_residuals.png")
    if os.path.exists(res_img):
        st.image(res_img, caption="Residual Diagnostics for Multiple Linear Regression (Test R^2 = 0.5186)", width=850)

# =============================================================================
# TAB 4: CUSTOMER SEGMENTATION
# =============================================================================
with tab4:
    st.subheader("Customer Segmentation: EM-Gaussian Mixture Model vs. Divisive Graph Clustering")
    st.write(
        "Unsupervised segmentation reveals natural clusters of borrowing behavior without utilizing the default label. "
        "Notice how both algorithms independently identify the same three core customer personas."
    )

    cluster_img = os.path.join(WORKSPACE_ROOT, "report", "clusters_comparison.png")
    if os.path.exists(cluster_img):
        st.image(cluster_img, caption="Comparison of EM-GMM and Divisive Graph (Girvan-Newman) Partitions", width=950)

    st.markdown("#### Financial Persona Profiles")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
            <h4>🟢 Prime Conservative (Cluster 2)</h4>
            <ul>
                <li><b>Default Rate:</b> ~21.7% (Lowest risk)</li>
                <li><b>Mean Loan Amount:</b> ~$1,387</li>
                <li><b>Mean Duration:</b> 12.3 months</li>
                <li><b>Mean Age:</b> 40.7 years</li>
                <li><b>Underwriting Strategy:</b> Fast-track approval; pre-approved micro-lines of credit.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
            <h4>🟡 Young Growth (Cluster 0)</h4>
            <ul>
                <li><b>Default Rate:</b> ~30.6% (Moderate risk)</li>
                <li><b>Mean Loan Amount:</b> ~$2,413</li>
                <li><b>Mean Duration:</b> 18.7 months</li>
                <li><b>Mean Age:</b> 29.4 years (Youngest)</li>
                <li><b>Underwriting Strategy:</b> Standard underwriting; credit limit graduation based on payment history.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
            <h4>🔴 High-Exposure Long-Term (Cluster 1)</h4>
            <ul>
                <li><b>Default Rate:</b> ~38.1% (Highest risk)</li>
                <li><b>Mean Loan Amount:</b> ~$6,322</li>
                <li><b>Mean Duration:</b> 33.2 months</li>
                <li><b>Mean Age:</b> 38.2 years</li>
                <li><b>Underwriting Strategy:</b> Require collateral / co-applicant; enforce strict debt-to-income caps.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 5: TRANSPARENT DECISION TREE RULES
# =============================================================================
with tab5:
    st.subheader("Explainable AI: Decision Tree If/Else Underwriting Rules")
    st.write(
        "Below are the hierarchical Gini decision rules extracted directly from the scratch Decision Tree classifier. "
        "Financial regulators require full auditability for credit decisions."
    )

    feature_names = data_dict["feature_names"]
    tree_rules = models_bundle["dt_model"].export_text(
        feature_names=feature_names, class_names=["Good Credit (0)", "Bad Credit (1)"]
    )

    st.code(tree_rules, language="text")
