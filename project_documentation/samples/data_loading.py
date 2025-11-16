
import pandas as pd

def load_and_explore_data(file_path, target_column):
    """
    Loads a dataset from a CSV file, prints its info, and shows the class distribution.

    Parameters:
    - file_path (str): The path to the CSV file.
    - target_column (str): The name of the target variable column.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    print(f"Loading data from: {file_path}")
    
    # Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

    print("\n--- Dataset Info ---")
    df.info()

    print("\n--- First 5 Rows ---")
    print(df.head())

    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the DataFrame.")
        return df

    print(f"\n--- Class Distribution of '{target_column}' ---")
    class_distribution = df[target_column].value_counts(normalize=True) * 100
    print(class_distribution)

    if class_distribution.min() > 20:
        print("\nWarning: The dataset may not be significantly imbalanced.")
    else:
        print("\nSuccess: The dataset is imbalanced and suitable for this project.")

    return df

if __name__ == '__main__':
    # This is an example of how to use the function.
    # You should replace 'path/to/your/data.csv' with the actual path to your dataset.
    # And 'target' with your actual target column name.
    
    # Create a dummy CSV for demonstration purposes
    dummy_data = {
        'feature1': range(100),
        'feature2': [i * 2 for i in range(100)],
        'target': [0] * 90 + [1] * 10  # 90% class 0, 10% class 1
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = 'dummy_imbalanced_data.csv'
    dummy_df.to_csv(dummy_csv_path, index=False)

    print("--- Running Data Loading Example ---")
    # Use the function to load and explore the dummy data
    df = load_and_explore_data(dummy_csv_path, 'target')

    if df is not None:
        print("\nData loading and exploration script finished successfully.")

