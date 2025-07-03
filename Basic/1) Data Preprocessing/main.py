import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_and_preprocess_data(file_path):
    """Load Iris dataset and check for required columns."""
    try:
        data = pd.read_csv(file_path)
        expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        if not all(col in data.columns for col in expected_columns):
            raise ValueError(
                "Dataset must contain columns: sepal_length, sepal_width, petal_length, petal_width, species")
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None


def train_and_evaluate(data):
    """Train and evaluate Logistic Regression model."""
    # Features and target
    x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = data['species']

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.2f}")

    return model, x.columns

def predict_new_sample(model, feature_names, sample):
    """Predict species for a new sample, ensuring feature names match."""
    # Convert sample to DataFrame with correct feature names
    sample_df = pd.DataFrame([sample], columns=feature_names)
    return model.predict(sample_df)[0]

def main():
    # Load data
    data = load_and_preprocess_data('iris.csv')
    if data is None:
        return

    # Train and evaluate
    model, feature_names = train_and_evaluate(data)

    # Example prediction
    sample = [5.1, 3.5, 1.4, 0.2]  # sepal_length, sepal_width, petal_length, petal_width
    predicted_species = predict_new_sample(model, feature_names, sample)
    print(f"Predicted species for sample {sample}: {predicted_species}")

if __name__ == "__main__":
    main()