"""
Data loading and preprocessing pipeline for German Credit dataset.
"""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "german_credit.csv")

CATEGORICAL_COLS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

NUMERICAL_COLS_ALL = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

TARGET_COL = "class"


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load raw German Credit dataset.
    Downloads from OpenML (data_id=31) if local CSV does not exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH) and not force_download:
        df = pd.read_csv(CSV_PATH)
    else:
        # Fetch from OpenML
        bunch = fetch_openml(data_id=31, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        df.to_csv(CSV_PATH, index=False)
    return df


class CreditDataPreprocessor:
    """
    Preprocessor that handles:
    - Target mapping: 'good' -> 0, 'bad' -> 1
    - One-hot encoding of categorical variables with known categories to ensure consistency.
    - Standardisation (zero mean, unit variance) of numeric variables using NumPy.
    - Prevents data leakage by fitting only on training data.
    """

    def __init__(self, target_regression: str = "credit_amount"):
        self.target_regression = target_regression
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = CATEGORICAL_COLS
        self.cat_categories_: Dict[str, List[str]] = {}
        self.encoded_feature_names_: List[str] = []
        self.num_means_: np.ndarray = np.array([])
        self.num_stds_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame):
        """Fit categories and numerical statistics on training DataFrame."""
        # Determine numerical columns excluding target classification and regression target if needed
        self.num_cols = [c for c in NUMERICAL_COLS_ALL if c in df.columns]

        # Fit categoricals
        self.cat_categories_ = {}
        for col in self.cat_cols:
            unique_vals = sorted(list(df[col].dropna().astype(str).unique()))
            self.cat_categories_[col] = unique_vals

        # Compute numerical means and stds using NumPy
        num_vals = df[self.num_cols].to_numpy(dtype=float)
        self.num_means_ = np.mean(num_vals, axis=0)
        self.num_stds_ = np.std(num_vals, axis=0)
        # Avoid division by zero
        self.num_stds_[self.num_stds_ == 0.0] = 1.0

        # Construct feature names
        feature_names = []
        for col in self.num_cols:
            feature_names.append(col)

        for col in self.cat_cols:
            for val in self.cat_categories_[col]:
                feature_names.append(f"{col}_{val}")

        self.encoded_feature_names_ = feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame to standardized numerical + one-hot encoded NumPy matrix."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")

        # Standardize numerical features
        num_vals = df[self.num_cols].to_numpy(dtype=float)
        num_standardized = (num_vals - self.num_means_) / self.num_stds_

        # One-hot encode categoricals
        ohe_blocks = []
        for col in self.cat_cols:
            cats = self.cat_categories_[col]
            series_str = df[col].astype(str).values
            # Matrix of shape (N, len(cats))
            col_block = np.zeros((len(df), len(cats)), dtype=float)
            for j, cat in enumerate(cats):
                col_block[:, j] = (series_str == cat).astype(float)
            ohe_blocks.append(col_block)

        cat_encoded = np.hstack(ohe_blocks)
        X = np.hstack([num_standardized, cat_encoded])
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_single_applicant(self, applicant_dict: Dict[str, Any]) -> np.ndarray:
        """Transform a single applicant dictionary into preprocessed feature vector (1, D)."""
        df_single = pd.DataFrame([applicant_dict])
        return self.transform(df_single)


def get_preprocessed_credit_data(
    test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Any]:
    """
    Full pipeline to load, split (stratified 80/20), and preprocess German Credit dataset.
    Returns:
        dict containing:
            'X_train', 'X_test', 'y_train', 'y_test' (classification: 0=good, 1=bad)
            'X_reg_train', 'X_reg_test', 'y_reg_train', 'y_reg_test' (regression for loan amount)
            'preprocessor': fitted preprocessor instance
            'feature_names': list of preprocessed feature names
            'raw_df': original raw dataframe
            'df_train_raw', 'df_test_raw': raw splits
    """
    df = load_raw_data()

    # Target: 0 = good, 1 = bad
    y = (df[TARGET_COL].astype(str).str.lower() == "bad").astype(int).to_numpy()

    # Stratified split 80/20
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=y, random_state=random_state
    )

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Classification features (all features)
    preprocessor = CreditDataPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)

    # Regression: Predict credit_amount
    # Target: raw loan amount
    y_reg_train = df_train["credit_amount"].to_numpy(dtype=float)
    y_reg_test = df_test["credit_amount"].to_numpy(dtype=float)

    # For regression features, exclude credit_amount
    reg_preprocessor = CreditDataPreprocessor(target_regression="credit_amount")
    # Custom fit excluding credit_amount from numerical features
    reg_preprocessor.cat_cols = CATEGORICAL_COLS
    reg_preprocessor.num_cols = [c for c in NUMERICAL_COLS_ALL if c != "credit_amount"]

    for col in reg_preprocessor.cat_cols:
        reg_preprocessor.cat_categories_[col] = sorted(
            list(df_train[col].dropna().astype(str).unique())
        )

    num_train = df_train[reg_preprocessor.num_cols].to_numpy(dtype=float)
    reg_preprocessor.num_means_ = np.mean(num_train, axis=0)
    reg_preprocessor.num_stds_ = np.std(num_train, axis=0)
    reg_preprocessor.num_stds_[reg_preprocessor.num_stds_ == 0.0] = 1.0

    reg_feat_names = list(reg_preprocessor.num_cols)
    for col in reg_preprocessor.cat_cols:
        for val in reg_preprocessor.cat_categories_[col]:
            reg_feat_names.append(f"{col}_{val}")
    reg_preprocessor.encoded_feature_names_ = reg_feat_names
    reg_preprocessor.is_fitted = True

    X_reg_train = reg_preprocessor.transform(df_train)
    X_reg_test = reg_preprocessor.transform(df_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_reg_train": X_reg_train,
        "X_reg_test": X_reg_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "preprocessor": preprocessor,
        "reg_preprocessor": reg_preprocessor,
        "feature_names": preprocessor.encoded_feature_names_,
        "reg_feature_names": reg_preprocessor.encoded_feature_names_,
        "raw_df": df,
        "df_train_raw": df_train,
        "df_test_raw": df_test,
    }


if __name__ == "__main__":
    data_dict = get_preprocessed_credit_data()
    print("X_train shape:", data_dict["X_train"].shape)
    print("X_test shape:", data_dict["X_test"].shape)
    print("y_train class counts:", np.bincount(data_dict["y_train"]))
    print("y_test class counts:", np.bincount(data_dict["y_test"]))
    print("X_reg_train shape:", data_dict["X_reg_train"].shape)
    print("Total features:", len(data_dict["feature_names"]))
