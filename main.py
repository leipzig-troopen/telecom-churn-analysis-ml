import argparse
import pandas as pd    
import src.config as config
from src.data_loader import load_data
from src.preprocess import clean_data, label_encode, feature_engineering_AvgPerMonth
from src.model_trainer import train_test_split_data, log_reg_model, random_forest_model, xgboost_model, evaluate_model

def main(data_path):
    # 1. Load Data
    df = load_data(data_path)

    # 2. Preprocess Data
    print("Preprocessing Data...")
    df_clean = clean_data(df)
    
    # Feature Engineering
    df_fe = feature_engineering_AvgPerMonth(df_clean)
    
    # Encoding 
    df_encoded, _ = label_encode(df_fe, config.CATEGORICAL_COLS)
    
    # 3. Split Data
    print("Splitting Data...")
    X = df_encoded.drop(columns=[config.TARGET,'gender', 'PhoneService'])
    y = df_encoded[config.TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # 4. Train & Evaluate Models
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

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Run Telco Churn pipeline')
   parser.add_argument('--data', type=str, default=config.RAW_DATA_PATH,
                         help=f'Path to the Telco CSV file (default: {config.RAW_DATA_PATH})')
   args = parser.parse_args()
   main(args.data)
