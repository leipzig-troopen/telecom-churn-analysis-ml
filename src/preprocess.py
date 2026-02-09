import pandas as pd
import src.config as config
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on the Telco dataset.

    - Drop `customerID` if present.
    - Convert `TotalCharges` to numeric and fill/coerce missing values.
    - Strip whitespace from object (string) columns.

    Returns the cleaned DataFrame.
    """
    df_clean = df.copy()

    # Drop customerID if exists
    if config.ID_COL in df_clean.columns:
        df_clean = df_clean.drop(columns=[config.ID_COL])

    # Strip whitespace from string columns
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Convert TotalCharges to numeric (some rows may be blank strings)
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    print(f"[CLEANING] Done. Data now has {df_clean.shape[1]} columns.")
    return df_clean

def impute_missing_values(df: pd.DataFrame, fill_value: float = None) -> Tuple[pd.DataFrame, float]:
    """
    Impute missing values in TotalCharges.
    If fill_value is None, calculates median from the dataframe (Training mode).
    If fill_value is provided, uses it (Inference/Test mode).
    """
    df_imputed = df.copy()
    if 'TotalCharges' in df_imputed.columns:
        if fill_value is None:
            fill_value = df_imputed['TotalCharges'].median()
        
        df_imputed['TotalCharges'] = df_imputed['TotalCharges'].fillna(fill_value)
    
    return df_imputed, fill_value


def label_encode(df: pd.DataFrame, categorical_cols: list, encoders=None):
    """
    Handle encoding for categorical data.
    - encoders=None: Training Mode (Fit & Transform)
    - encoders=dict: Inference Mode (Transform Only)
    """
    df_clean = df.copy()
    if encoders is None:
        # TRAINING PHASE: Learn patterns (Fit)
        encoders = {'label_encoder': {}, 'oh_encoder': None, 'oh_cols': []}
        
        # 1. Label Encoding for Binary (2 Categories)
        binary_cols = [col for col in categorical_cols if df_clean[col].nunique() <= 2]
        for col in binary_cols:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            encoders['label_encoder'][col] = le
        
        # 2. One-Hot Encoding for Multi-category (>2 Categories)
        multi_cols = [col for col in categorical_cols if df_clean[col].nunique() > 2]
        if multi_cols:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
            ohe_data = ohe.fit_transform(df_clean[multi_cols])
            ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(multi_cols), index=df_clean.index)
            df_encoded = pd.concat([df_clean.drop(columns=multi_cols), ohe_df], axis=1)
            
            encoders['oh_encoder'] = ohe
            encoders['oh_cols'] = multi_cols
        else:
            df_encoded = df_clean.copy()
        
        print(f"[ENCODING] done: {len(binary_cols)} binary columns label-encoded, {len(multi_cols)} multi-category columns one-hot encoded.")
        return df_encoded, encoders
    
    else:
        # INFERENCE PHASE: Follow previous patterns (Transform Only)
        # 1. Apply Label Encoding
        for col, le in encoders['label_encoder'].items():
            df_clean[col] = le.transform(df_clean[col])
        
        # 2. Apply One-Hot Encoding
        ohe = encoders['oh_encoder']
        multi_cols = encoders['oh_cols']
        if ohe and multi_cols:
            ohe_data = ohe.transform(df_clean[multi_cols])
            ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(multi_cols), index=df_clean.index)
            df_encoded = pd.concat([df_clean.drop(columns=multi_cols), ohe_df], axis=1)
        else:
            df_encoded = df_clean.copy()
        
        print(f"[ENCODING] done: Applied label encoding and one-hot encoding using existing encoders.")
        return df_encoded, encoders


def scale_features(df: pd.DataFrame, features: list, scaler=None) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Scale numerical features.
    
    Args:
        df: Input DataFrame
        features: List of numeric columns to scale
        scaler: StandardScaler object (if None, fit new scaler)
    
    Returns:
        Tuple[pd.DataFrame, Optional[StandardScaler]]: Scaled data and scaler object
    """
    df_scaled = df.copy()
    
    if scaler is None:
        # Training mode: fit & transform
        scaler = StandardScaler()
        df_scaled[features] = scaler.fit_transform(df_scaled[features])
        return df_scaled, scaler
    else:
        # Inference mode: transform only
        df_scaled[features] = scaler.transform(df_scaled[features])
        return df_scaled, scaler  # Return scaler also for consistency


def feature_engineering_TenureGroup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the Telco dataset.
    
    - Create `TenureGroup` based on `tenure`.
    
    Returns the DataFrame with new features.
    """
    df_fe = df.copy()
    
    # Create tenure groups
    bins = [0, 12, 24, 48, 60, 72]
    labels = ['0-12', '13-24', '25-48', '49-60', '61-72']
    df_fe['TenureGroup'] = pd.cut(df_fe['tenure'], bins=bins, labels=labels, right=True)
    
    print("[FEATURE ENGINEERING] done: Created 'TenureGroup' feature.")
    return df_fe


def feature_engineering_AvgPerMonth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create AverageChargesPerMonth feature.
    
    Returns the DataFrame with new feature.
    """
    df_fe = df.copy()
    
    df_fe['AvgChargesPerMonth'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)  # +1 to avoid division by zero
    print("[FEATURE ENGINEERING] done: Created 'AvgChargesPerMonth' feature.")
    return df_fe
