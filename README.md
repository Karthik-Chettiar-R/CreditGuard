# CreditGuard: Credit-Risk & Customer Segmentation Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy From Scratch](https://img.shields.io/badge/Algorithms-100%25%20NumPy-orange.svg)](https://numpy.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)](https://docs.pytest.org/)

CreditGuard is an end-to-end, production-grade financial risk assessment and customer segmentation platform designed to solve a foundational lending problem: **quantifying borrower credit default probability, determining safe capital exposure limits, and profiling customer cohorts into risk personas.**

All core algorithms are implemented **from scratch using only NumPy** without relying on Scikit-Learn model implementations. Scikit-Learn is used solely as an independent benchmark harness for test comparisons and data splitting.

---

## 🏦 Business Problem Statement

Retail and commercial lenders face an asymmetric payoff matrix:
1. **False Positive (Type I Error)**: Flagging a creditworthy borrower as high-risk causes an opportunity cost of forgone interest spread (~$250 – $600).
2. **False Negative (Type II Error)**: Approving a borrower who subsequently defaults results in catastrophic capital loss, often forfeiting 100% of the loan principal ($2,000 – $6,000+).

Because default losses dwarf marginal interest margins by an order of magnitude, **Recall on the "Bad Credit" class is the paramount operational metric**. A model that merely achieves high overall accuracy by prioritizing the majority "Good Credit" class (70% base rate) is financially dangerous.

Furthermore, capital allocation requires **Customer Segmentation**: grouping applicants into distinct behavioral personas (e.g. *Prime Conservative*, *Young Growth*, *High-Exposure Long-Term*) to tailor interest rate premiums and loan duration caps.

---

## 📐 Algorithmic Suite (Implemented from Scratch in NumPy)

All seven algorithms reside in dedicated modules under `src/`:

| Algorithm | File | Mathematical Formulation & Scratch Technique |
| :--- | :--- | :--- |
| **Simple & Multiple Linear Regression** | [`src/linear_regression.py`](src/linear_regression.py) | **Simple:** Univariate closed form $\beta_1 = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}$, $\beta_0 = \bar{y} - \beta_1 \bar{x}$.<br>**Multiple:** Exact Normal Equation $(X^T X + \lambda I)^{-1} X^T y$ with pseudoinverse fallback + Batch Gradient Descent with loss convergence tracking. |
| **Principal Component Analysis (PCA)** | [`src/pca.py`](src/pca.py) | Sample covariance matrix $\Sigma = \frac{1}{N-1} X_c^T X_c$, symmetric eigendecomposition via `np.linalg.eigh`, descending eigenvalue sorting, deterministic sign-flip convention, explained variance ratios, and orthogonal subspace projection. |
| **Decision Tree Classifier** | [`src/decision_tree.py`](src/decision_tree.py) | Weighted Gini impurity minimization $I_G(S) = 1 - \sum p_k^2$, recursive tree building with `max_depth` and `min_samples_split`, and readable `export_text()` / `print_tree()` generating hierarchical if/else rules. |
| **AdaBoost Classifier** | [`src/adaboost.py`](src/adaboost.py) | Adaptive boosting with Gini decision stumps (depth-1 trees), sample weight reweighting $w_i \leftarrow w_i \exp(-\alpha_m y_i h_m(x_i))$, estimator weights $\alpha_m = \frac{1}{2} \ln \frac{1-\epsilon_m}{\epsilon_m}$, and sigmoid-calibrated probability estimation. |
| **Support Vector Machine (Linear + Kernel)** | [`src/svm.py`](src/svm.py) | **Linear:** Soft-margin hinge loss optimization via mini-batch subgradient descent $\min \frac{1}{2}\|w\|^2 + C \sum \max(0, 1 - y_i(w^T x_i + b))$.<br>**Bonus Kernel:** Non-linear RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ trained with Simplified Sequential Minimal Optimization (SMO). |
| **EM Gaussian Mixture Model (GMM)** | [`src/gmm.py`](src/gmm.py) | Expectation-Maximization with full covariance matrices, K-means++ initialization, Cholesky decomposition for numerical stability, log-sum-exp density evaluation, monotonic log-likelihood tracking, and Bayesian Information Criterion (BIC). |
| **Divisive Graph Clustering** | [`src/graph_clustering.py`](src/graph_clustering.py) | Symmetric $k$-NN customer similarity graph on PCA space, top-down Girvan-Newman community splitting using Brandes BFS edge betweenness centrality, and Newman-Girvan modularity optimization. |

---

## 📊 Benchmark Results

### 1. Loan Amount Regression Benchmark (Test Set $N=200$)
Predicting loan principal amount ($y = \text{credit\_amount}$) based on applicant financial profile:
- **Scratch Multiple Linear Regression (Normal Equation)**: Test $R^2 = \mathbf{0.5186}$, Test $\text{RMSE} = \mathbf{\$2,312.79}$
- **Scikit-Learn `LinearRegression`**: Test $R^2 = \mathbf{0.5186}$, Test $\text{RMSE} = \mathbf{\$2,312.79}$ (exact numerical agreement to 5 significant digits)
- **Scratch Simple Linear Regression (Duration only)**: Test $R^2 = \mathbf{0.3393}$ (demonstrates that loan duration alone accounts for ~34% of variance, while full financial attributes explain 52%).

### 2. Credit Default Classification Benchmark (Stratified 80/20 Holdout)

| Model | Feature Representation | Accuracy | Precision (Bad Credit) | Recall (Bad Credit) [KEY] | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost (Scratch)** | **Raw (61 cols)** | **77.0%** | **0.659** | **0.483** | **0.558** | **0.804** |
| Linear SVM (Scratch) | Raw (61 cols) | 76.5% | 0.741 | 0.333 | 0.460 | 0.794 |
| Decision Tree (Scratch) | Raw (61 cols) | 67.0% | 0.453 | 0.483 | 0.468 | 0.686 |
| **AdaBoost (Sklearn)** | Raw (61 cols) | 72.0% | 0.548 | 0.383 | 0.451 | 0.767 |
| Linear SVM (Sklearn) | Raw (61 cols) | 77.5% | 0.642 | 0.567 | 0.602 | 0.799 |
| Decision Tree (Sklearn) | Raw (61 cols) | 66.0% | 0.439 | 0.483 | 0.460 | 0.691 |
| **AdaBoost (Scratch)** | **PCA (10 comps)** | **73.0%** | **0.579** | **0.367** | **0.449** | **0.732** |
| Linear SVM (Scratch) | PCA (10 comps) | 76.5% | 0.760 | 0.317 | 0.447 | 0.761 |
| Decision Tree (Scratch) | PCA (10 comps) | 72.0% | 0.559 | 0.317 | 0.404 | 0.705 |

#### 🔑 Key Insight on Recall & Decision Thresholding
At the default 0.50 probability cutoff, AdaBoost achieves 48.3% Bad-Credit Recall with 65.9% Precision and an ROC-AUC of 0.804. By calibrating the decision threshold to **0.35** in the CreditGuard underwriting dashboard, **Bad-Credit Recall rises to >75%**, capturing three out of four defaulting borrowers and drastically reducing loan portfolio write-offs.

---

## 👥 Customer Segmentation & Persona Profiles

Running unsupervised **EM-GMM** and **Divisive Graph Clustering (Girvan-Newman)** on PCA-reduced representations independently uncovered the identical 3-segment market structure:

| Segment Persona | Clustering Method | Cohort % | Mean Loan Amount | Mean Duration | Mean Age | Default Rate | Underwriting Strategy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Prime Conservative** | EM-GMM (Cluster 2)<br>Graph (Cluster 1) | ~25–30% | **$1,387 – $1,424** | **12.3 – 12.5 mo** | **40.5 – 40.7 yr** | **21.7% – 23.2%** | **Low Risk.** Pre-approved micro-lines, fast-track automated approval, prime interest rates. |
| **Young Growth** | EM-GMM (Cluster 0)<br>Graph (Cluster 2) | ~40–43% | **$2,191 – $2,413** | **17.1 – 18.7 mo** | **29.0 – 29.4 yr** | **28.6% – 30.6%** | **Moderate Risk.** Standard underwriting, initial conservative credit limits, limit increases tied to timely payments. |
| **High-Exposure Long-Term** | EM-GMM (Cluster 1)<br>Graph (Cluster 0) | ~28–36% | **$5,347 – $6,322** | **29.7 – 33.2 mo** | **38.2 – 39.8 yr** | **35.5% – 38.1%** | **High Risk.** Strict debt-to-income caps, mandatory collateral or solvent co-applicants, risk-based pricing. |

---

## 📁 Project Directory Structure

```text
creditGuard/
│
├── data/
│   └── german_credit.csv              # Local cached copy of UCI Statlog German Credit Data
│
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── data_loader.py                 # Data fetching, OHE, standardisation, stratified split
│   ├── linear_regression.py           # Simple & Multiple Linear Regression (Normal Eq + GD)
│   ├── pca.py                         # Covariance matrix + eigendecomposition PCA
│   ├── decision_tree.py               # Gini Decision Tree with rule exporter/printer
│   ├── adaboost.py                    # AdaBoost with Gini decision stumps & sample reweighting
│   ├── svm.py                         # Soft-margin subgradient Linear SVM + SMO Kernel SVM
│   ├── gmm.py                         # EM Gaussian Mixture Model with monotonic LL & BIC
│   ├── graph_clustering.py            # kNN + Girvan-Newman edge betweenness clustering
│   └── pipeline.py                    # End-to-end evaluation & artifact generation runner
│
├── tests/
│   ├── test_linear_regression.py      # Numerical equality tests vs sklearn LinearRegression
│   ├── test_pca.py                    # Variance ratio & projection tests vs sklearn PCA
│   ├── test_decision_tree.py          # Gini, splits, rule generation & sklearn comparisons
│   ├── test_adaboost.py               # Boosting error minimization & sklearn comparison
│   ├── test_svm.py                    # Hinge loss minimization, Linear & Kernel SVM tests
│   ├── test_gmm.py                    # Asserts non-decreasing log-likelihood & BIC tests
│   └── test_graph_clustering.py       # Graph topology, Girvan-Newman splits & modularity
│
├── notebooks/
│   └── creditguard_pipeline.ipynb     # Interactive Jupyter notebook walkthrough
│
├── report/
│   ├── regression_residuals.png       # Residual vs predicted & error distribution plot
│   ├── pca_scree_plot.png             # Scree bar chart & cumulative explained variance
│   ├── pca_2d_scatter.png             # 2D PCA projection colored by default label
│   ├── roc_curves.png                 # ROC curves for all scratch classifiers
│   ├── clusters_comparison.png        # EM-GMM vs Divisive Graph clustering partitions
│   └── classification_benchmark.csv   # Comprehensive benchmark results table
│
├── app.py                             # Interactive Streamlit application
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation, architecture & operational guide
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/your-username/creditGuard.git
cd creditGuard
pip install -r requirements.txt
```

### 2. Running Unit Tests
Execute the complete test suite. All tests verify scratch implementations against analytical solutions or Scikit-Learn baselines, and verify monotonic log-likelihood on EM:
```bash
python -m pytest -v
```
*Result: 17/17 tests passing in ~2 seconds.*

### 3. Running the End-to-End Pipeline
To re-run data ingestion, model benchmarks, and generate visual charts in `report/`:
```bash
python src/pipeline.py
```

### 4. Launching the Interactive Streamlit Web App
Launch the financial dashboard locally:
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🖥️ Streamlit Application Walkthrough

The web app is structured into five interactive modules:
1. **Applicant Risk Assessment**:
   - Dynamic underwriting form for loan terms, checking balance, credit history, employment, and demographics.
   - Calculates **Default Probability**, **Risk Classification** (Low, Moderate, High), **Suggested Safe Loan Cap** (based on Multiple Linear Regression haircut), and **Assigned Customer Segment**.
2. **Model Benchmarks & Comparison**:
   - Side-by-side performance table (Accuracy, Precision, Recall, F1, ROC-AUC) comparing scratch models with Scikit-Learn.
   - Interactive bar charts and holdout ROC curves.
3. **PCA & Risk Space**:
   - Scree plot of cumulative explained variance.
   - 2D customer risk scatter plot with residual error diagnostics.
4. **Customer Segmentation**:
   - Side-by-side visual comparison of EM-GMM vs Divisive Graph partitions.
   - Strategic underwriting recommendations for each borrowing persona.
5. **Explainable AI (Decision Tree Rules)**:
   - Expandable viewer showing transparent, auditable if/else decision logic derived from the scratch tree.

---

## 📜 License
This project is open-source under the MIT License.
