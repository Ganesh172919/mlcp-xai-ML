
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

def train_baseline_model(df, target_column, features=None):
    """
    Trains a baseline Logistic Regression model on an imbalanced dataset and evaluates it.

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

    print("--- Training Baseline Model ---")

    # 1. Define Features (X) and Target (y)
    if features:
        X = df[features]
    else:
        X = df.drop(target_column, axis=1)
    
    y = df[target_column]

    # 2. Train-Test Split
    # We use stratify=y to ensure the class distribution is the same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 3. Initialize and Train the Model
    print("\nTraining Logistic Regression model on the original, imbalanced data...")
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Make Predictions and Evaluate
    print("\n--- Baseline Model Evaluation on Test Set ---")
    predictions = model.predict(X_test)

    print("\nClassification Report:")
    # Note the poor recall and f1-score for the minority class (class 1)
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    bal_acc = balanced_accuracy_score(y_test, predictions)
    print(f"\nBalanced Accuracy: {bal_acc:.2f}")
    
    print("\nNotice how accuracy might be high, but recall for the minority class is likely low.")
    print("This is the problem we aim to solve with oversampling.")
    
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

    print("--- Running Baseline Model Training Example ---")
    baseline_model = train_baseline_model(dummy_df, 'target')

    if baseline_model:
        print("\nBaseline model script finished successfully.")
