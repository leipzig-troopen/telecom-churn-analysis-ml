import src.config as config
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from scipy.stats import randint, uniform


def train_test_split_data(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    X (DataFrame): Features.
    y (Series): Target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    X_train, X_test, y_train, y_test: Split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    return X_train, X_test, y_train, y_test

def log_reg_model(X_train, y_train):
    """
    Trains a Logistic Regression model.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained LogisticRegression model.
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model

def random_forest_model(X_train, y_train):
    """
    Trains a Random Forest Classifier.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=config.RANDOM_STATE,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def xgboost_model(X_train, y_train):
    """
    Trains an XGBoost Classifier.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained XGBClassifier model.
    """
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=config.RANDOM_STATE,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model


def lightgbm_model(X_train, y_train):
    """
    Trains a LightGBM Classifier.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained LGBMClassifier model.
    """
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=config.RANDOM_STATE,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Handles prediction and evaluation in one place to keep main.py cleaner.
    """
    # 1. Get label predictions (0/1)
    y_pred = model.predict(X_test)
    
    # 2. Get probabilities (decimal) specifically for AUC & AP
    y_proba = model.predict_proba(X_test)[:, 1] 

    print(f"\n{'=' *60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'=' *60}")
    
    # Classification Report using labels (0/1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # AUC-ROC & AP use probabilities for accuracy
    auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    
    print(f"AUC-ROC Score: {auc:.3f}")
    print(f"Average Precision: {avg_prec:.3f}")
    print(f"{'=' *60}")



def train_xgb_tuned(X_train, y_train):
    """
    Runs Hyperparameter Tuning on XGBoost using GridSearchCV.
    Focuses on improving Recall to detect Churn.
    """
    
    # 1. Initialize base model
    xgb = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=config.RANDOM_STATE
    )
    
    # 2. Define Parameter Grid
    # scale_pos_weight helps the model be more sensitive to the Churn class
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'min_child_weight':randint(1, 6),
        'gamma': uniform(0, 0.5),
        'learning_rate': uniform(0.01, 0.2),
        'scale_pos_weight': uniform(2.5, 1.5),
        'subsample': uniform(0.8, 0.2),
        'colsample_bytree': uniform(0.8, 0.2)
    }
    
    # 3. Setup RandomizedSearchCV
    # Using 'f1' scoring to balance Precision and Recall
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=50,  # Number of parameter settings sampled
        scoring='f1',
        cv=5,
        verbose=0,
        random_state=config.RANDOM_STATE,
        n_jobs=3
    )
    
    # 4. Fitting Model
    print("--- Starting XGBoost Hyperparameter Tuning ---")
    random_search.fit(X_train, y_train)
    
    # 5. Get best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"Tuning Complete. Best Params: {best_params}")
    return best_model, best_params

def train_rf_tuned(X_train, y_train):
    """
    Runs Hyperparameter Tuning on Random Forest using GridSearchCV.
    Optimized to handle data imbalance in the Telco Churn project.
    """
    
    # 1. Initialize base model
    rf = RandomForestClassifier(bootstrap=True, random_state=config.RANDOM_STATE)
    
    # 2. Define Parameter Grid
    # class_weight='balanced' is key to increasing Recall
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9],
        'max_depth': randint(5, 25),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'class_weight': ['balanced', 'balanced_subsample'] 
    }
    
    # 3. Setup RandomizedSearchCV
    # n_jobs=3 to keep system responsive
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1', # Optimizing balance between Precision & Recall
        cv=5,
        verbose=0,
        random_state=config.RANDOM_STATE,
        n_jobs=3 
    )
    
    # 4. Fitting Model
    print("--- Starting Random Forest Hyperparameter Tuning ---")
    random_search.fit(X_train, y_train)
    
    # 5. Output best results
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"Tuning Complete. Best Params: {best_params}")
    return best_model, best_params  


def train_lgbm_tuned(X_train, y_train):
    """
    Runs Hyperparameter Tuning on LightGBM using GridSearchCV.
    Focuses on improving Recall to better identify Churn cases.
    """
    
    lgbm = lgb.LGBMClassifier(
        random_state=config.RANDOM_STATE,
        bagging_seed=config.RANDOM_STATE,
        feature_fraction_seed=config.RANDOM_STATE
        )

    param_dist = {
        'n_estimators': randint(100, 1000),
        'num_leaves': randint(20, 150),
        'max_depth': randint(5, 20),
        'learning_rate': uniform(0.01, 0.2),
        'min_child_samples': randint(5, 50),
        'bagging_fraction': uniform(0.8, 0.2),
        'feature_fraction': uniform(0.8, 0.2),
        'bagging_freq': [1, 3, 5],
        'scale_pos_weight': uniform(2.5, 1.5),
    }

    grid_search = RandomizedSearchCV(
        estimator=lgbm, # type: ignore
        param_distributions=param_dist,
        scoring='f1',
        n_iter=50,
        cv=5,
        verbose=0,
        random_state=config.RANDOM_STATE,
        n_jobs=3
    )

    print("--- Starting LightGBM Hyperparameter Tuning ---")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Tuning Complete. Best Params: {best_params}")
    return best_model, best_params