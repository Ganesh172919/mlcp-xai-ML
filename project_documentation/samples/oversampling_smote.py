
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter

def train_model_with_smote(df, target_column, features=None):
    """
    Trains a Logistic Regression model on a dataset balanced with SMOTE and evaluates it.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The name of the target variable column.
    - features (list, optional): A list of feature column names. 
                                 If None, all columns except the target are used.

    Returns:
    - model: The trained Logistic Regression model.
    """
    if df is None or target_column not in df.columns:
        print("Error: DataFrame is invalid or target column not found.")
        return None

    print("--- Training Model with SMOTE Oversampling ---")

    # 1. Define Features (X) and Target (y)
    if features:
        X = df[features]
    else:
        X = df.drop(target_column, axis=1)
    
    y = df[target_column]

    # 2. Train-Test Split (BEFORE oversampling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nOriginal training set distribution: {Counter(y_train)}")

    # 3. Apply SMOTE to the training data ONLY
    print("Applying SMOTE to the training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Resampled training set distribution: {Counter(y_train_resampled)}")

    # 4. Initialize and Train the Model on the Resampled Data
    print("\nTraining Logistic Regression model on the new, balanced data...")
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train_resampled, y_train_resampled)
    print("Model training complete.")

    # 5. Make Predictions and Evaluate on the original, untouched test set
    print("\n--- SMOTE Model Evaluation on Test Set ---")
    predictions = model.predict(X_test)

    print("\nClassification Report:")
    # Compare this report to the baseline model's report.
    # Recall and F1-score for the minority class should be much better.
    print(classification_report(y_test, predictions))

    bal_acc = balanced_accuracy_score(y_test, predictions)
    print(f"\nBalanced Accuracy: {bal_acc:.2f}")
    
    print("\nCompare these results to the baseline. The improvement in minority class metrics demonstrates the power of oversampling.")
    
    return model

if __name__ == '__main__':
    # Create a dummy imbalanced DataFrame for demonstration
    dummy_data = {
        'feature1': range(1000),
        'feature2': [i * 1.5 for i in range(1000)],
        'feature3': [i % 100 for i in range(1000)],
        'target': [0] * 900 + [1] * 100  # 90% class 0, 10% class 1
    }
    dummy_df = pd.DataFrame(dummy_data)

    print("--- Running SMOTE Model Training Example ---")
    smote_model = train_model_with_smote(dummy_df, 'target')

    if smote_model:
        print("\nSMOTE model script finished successfully.")
