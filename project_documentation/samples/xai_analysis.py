
import pandas as pd
import shap
import xgboost
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def perform_xai_analysis(df, target_column, features=None):
    """
    Trains an XGBoost model on SMOTE-balanced data and performs SHAP analysis.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the target variable column.
    - features (list, optional): A list of feature column names. 
                                 If None, all columns except the target are used.
    """
    if df is None or target_column not in df.columns:
        print("Error: DataFrame is invalid or target column not found.")
        return

    print("--- Performing XAI Analysis with SHAP ---")

    # 1. Define Features (X) and Target (y)
    if features:
        X = df[features]
    else:
        X = df.drop(target_column, axis=1)
    
    y = df[target_column]

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 4. Train a more complex model (like XGBoost) for XAI
    print("\nTraining XGBoost model on balanced data...")
    # Using eval_set to prevent verbose output during training
    model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_resampled, y_train_resampled, verbose=False)
    print("Model training complete.")

    # 5. Perform SHAP Analysis
    print("\nCalculating SHAP values... This may take a moment.")
    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("SHAP values calculated.")

    # 6. Generate and Save Visualizations
    print("\nGenerating SHAP summary plot...")
    # The summary plot gives a global view of feature importance
    shap.summary_plot(shap_values, X_test, show=False)
    # In a real script, you would save this plot to a file
    # plt.savefig('shap_summary_plot.png')
    # plt.close()
    print("SHAP summary plot generated. In a real script, you would see this plot or save it to a file.")
    print("This plot shows which features are most important and their impact on the model's output.")

    print("\nGenerating SHAP force plot for a single prediction...")
    # The force plot explains a single prediction
    # We'll explain the first instance in the test set
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], show=False)
    # shap.save_html('shap_force_plot.html', force_plot)
    print("SHAP force plot generated. This would be saved as an HTML file in a real application.")
    print("It shows how each feature contributed to a specific prediction.")
    
    print("\n\nXAI analysis provides deep insights into *why* the model makes its decisions.")
    print("By analyzing the SHAP values, you can ensure the model is learning meaningful patterns and not relying on artifacts.")


if __name__ == '__main__':
    # Create a dummy imbalanced DataFrame for demonstration
    dummy_data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 100,
        'income': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140] * 100,
        'credit_score': [600, 650, 700, 750, 800, 580, 620, 680, 720, 780] * 100,
        'target': [0] * 900 + [1] * 100  # 90% class 0, 10% class 1
    }
    dummy_df = pd.DataFrame(dummy_data)

    print("--- Running XAI Analysis Example ---")
    perform_xai_analysis(dummy_df, 'target')

    print("\nXAI analysis script finished successfully.")
