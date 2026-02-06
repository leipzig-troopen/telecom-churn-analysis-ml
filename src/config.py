# ==========================================
# PATH CONFIGURATION
# ==========================================
RAW_DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_EXPORT_PATH = "models/"

# ==========================================
# DATA FEATURES
# ==========================================
TARGET = "Churn"

# Columns to be dropped as they are irrelevant for prediction
ID_COL = "customerID"

# Numerical columns (to be scaled)
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Categorical columns (to be encoded)
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", 
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
    "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod", "Churn"
]


# ==========================================
# MODEL SETTINGS
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.2