import argparse
import pandas as pd    
import src.config as config
from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_data
from src.preprocess import clean_data, label_encode, feature_engineering_AvgPerMonth, impute_missing_values
from src.model_trainer import (
    train_test_split_data,
    log_reg_model,
    random_forest_model,
    xgboost_model,
    train_xgb_tuned,
    train_rf_tuned,
    evaluate_model
)

def main(data_path):
    # 1. Load Data
    df = load_data(data_path)

    # 2. Preprocess Data
    print("Preprocessing Data...")
    df_clean = clean_data(df)
    
    # 3. Split Data (EARLY SPLIT TO PREVENT LEAKAGE)
    print("Splitting Data...")
    # Drop target and columns we don't want as features
    X = df_clean.drop(columns=[config.TARGET, 'gender', 'PhoneService'])
    y = df_clean[config.TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # 4. Impute Missing Values (Fit on Train, Transform Test)
    print("Imputing Missing Values...")
    X_train, median_val = impute_missing_values(X_train)
    X_test, _ = impute_missing_values(X_test, fill_value=median_val)

    # 5. Feature Engineering
    X_train = feature_engineering_AvgPerMonth(X_train)
    X_test = feature_engineering_AvgPerMonth(X_test)

    # 6. Encoding Features
    # Filter categorical cols to ensure they exist in X (exclude Target)
    feat_cols = [c for c in config.CATEGORICAL_COLS if c in X_train.columns]
    X_train, encoders = label_encode(X_train, feat_cols)
    X_test, _ = label_encode(X_test, feat_cols, encoders)

    # 7. Encode Target (y) separately
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)

    # 8. Train & Evaluate Models
    models = {
        "Logistic Regression": log_reg_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgboost_model
    }

    for name, model_func in models.items():
        print(f"\nTraining {name}...")
        model = model_func(X_train, y_train)
        print(f"Evaluating {name}...")
        evaluate_model(model, X_test, y_test, model_name=name)

    models_tune = {
        "Random Forest": train_rf_tuned,
        "XGBoost": train_xgb_tuned
    }

    for name, model_func in models_tune.items():
        print(f"\nTuning {name}...")
        model, _ = model_func(X_train, y_train)
        print(f"Evaluating Tuned {name}...")
        evaluate_model(model, X_test, y_test, model_name=f"Tuned {name}")


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Run Telco Churn pipeline')
   parser.add_argument('--data', type=str, default=config.RAW_DATA_PATH,
                         help=f'Path to the Telco CSV file (default: {config.RAW_DATA_PATH})')
   args = parser.parse_args()
   main(args.data)
