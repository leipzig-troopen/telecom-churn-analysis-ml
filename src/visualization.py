import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix

def plot_churn_distribution(df):
    """Plot Churn target distribution."""
    df_plot = df.copy()
    # Ensure Churn labels are 'Yes'/'No' for plotting
    if df_plot['Churn'].dtype != object:
        df_plot['Churn'] = df_plot['Churn'].map({1: 'Yes', 0: 'No'})

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        x=df_plot['Churn'].value_counts().index,
        y=df_plot['Churn'].value_counts(normalize=True).values * 100
    )

    plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Churn Status', fontsize=12)
    plt.ylabel('Percentage of Customers', fontsize=12)
    plt.ylim(0, 110)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=11, fontweight='bold')
    plt.show()

def plot_numerical_features(df, features):
    """Plot numerical feature distribution against Churn."""
    df_plot = df.copy()
    # Normalize Churn labels
    if df_plot['Churn'].dtype != object:
        df_plot['Churn'] = df_plot['Churn'].map({1: 'Yes', 0: 'No'})

    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(features, 1):
        plt.subplot(1, len(features), i)
        sns.kdeplot(df_plot[df_plot['Churn'] == 'No'][feature].dropna(), label='No Churn')
        sns.kdeplot(df_plot[df_plot['Churn'] == 'Yes'][feature].dropna(), label='Churn')
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

def plot_categorical_features(df, features):
    """Plot categorical features against Churn."""
    df_plot = df.copy()
    if df_plot['Churn'].dtype != object:
        df_plot['Churn'] = df_plot['Churn'].map({1: 'Yes', 0: 'No'})

    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=feature, hue='Churn', data=df_plot)
        plt.title(f'Customer Churn Distribution by {feature}')
        plt.xticks(rotation=45)
        plt.show()
    
def plot_correlation_heatmap(df, numerical_features):
    """Plot correlation heatmap between numerical features."""
    plt.figure(figsize=(8, 4))
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap='RdBu', fmt='.2f')
    plt.title('Numerical Features Correlation Heatmap')
    plt.show()

def plot_outliers(df, numerical_features):
    """Plot boxplot to detect outliers in numerical features."""
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(1, len(numerical_features), i)
        sns.boxplot(y=df[feature])
        plt.title(f'Outliers in {feature}')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix for model prediction results."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_shap_summary(model, X_train, model_name):
    """Displays SHAP Summary Plot for global model interpretation."""
    # SHAP usually works very well with tree-based models (RF/XGBoost)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    plt.figure(figsize=(6, 5))
    plt.title(f"SHAP Summary Plot - {model_name}")
    shap.summary_plot(shap_values, X_train, show=False)
    plt.show()
   