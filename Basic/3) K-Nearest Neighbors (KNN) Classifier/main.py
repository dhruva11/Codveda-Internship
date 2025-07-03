import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


def load_and_preprocess_data(file_path):
    """Load and preprocess Stock Prices dataset."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in the current directory: {os.getcwd()}")
        return None
    try:
        data = pd.read_csv(file_path)
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Volume', 'Close']
        columns = data.columns.str.lower()
        required_cols = {col.lower(): col for col in data.columns}
        missing_cols = [col for col in expected_columns if col.lower() not in columns]
        if missing_cols:
            print(
                f"Error: Dataset '{file_path}' must contain columns: {expected_columns}. Missing: {missing_cols}. Found: {list(data.columns)}")
            return None

        # Rename columns for consistency
        rename_dict = {required_cols[col.lower()]: col for col in expected_columns if col.lower() in columns}
        data = data.rename(columns=rename_dict)

        # Handle missing values
        data = data.dropna(subset=['Open', 'High', 'Low', 'Volume', 'Close'])

        return data
    except Exception as e:
        print(f"Error loading dataset '{file_path}': {str(e)}")
        return None


def train_and_evaluate(data):
    """Train and evaluate Linear Regression model."""
    features = ['Open', 'High', 'Low', 'Volume']
    X = data[features]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Regression RMSE: {rmse:.2f}")

    return model, features


def predict_new_sample(model, feature_names, sample):
    """Predict closing price for a new sample."""
    sample_df = pd.DataFrame([sample], columns=feature_names)
    return model.predict(sample_df)[0]


def main():
    file_path = 'Stock Prices Data Set.csv'
    data = load_and_preprocess_data(file_path)
    if data is None:
        return

    model, feature_names = train_and_evaluate(data)

    sample = [100.5, 102.0, 99.5, 1000000]  # Open, High, Low, Volume
    predicted_price = predict_new_sample(model, feature_names, sample)
    print(f"Predicted closing price for sample {sample}: {predicted_price:.2f}")


if __name__ == "__main__":
    main()