import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def preprocess_text(text):
    """Clean text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def load_and_preprocess_data(file_path):
    """Load and preprocess Sentiment dataset."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in the current directory: {os.getcwd()}")
        return None
    try:
        data = pd.read_csv(file_path)
        # Case-insensitive column check
        columns = data.columns.str.lower()
        text_col = None
        sentiment_col = None
        for col in columns:
            if col == 'text':
                text_col = data.columns[columns == col][0]
            if col == 'sentiment':
                sentiment_col = data.columns[columns == col][0]

        if text_col is None or sentiment_col is None:
            print(
                f"Error: Dataset '{file_path}' must contain 'Text' and 'Sentiment' columns (case-insensitive). Found columns: {list(data.columns)}")
            return None

        # Rename columns for consistency
        data = data.rename(columns={text_col: 'text', sentiment_col: 'sentiment'})

        # Handle missing or invalid text
        data['text'] = data['text'].fillna("").apply(preprocess_text)
        data = data.dropna(subset=['sentiment'])

        return data
    except Exception as e:
        print(f"Error loading dataset '{file_path}': {str(e)}")
        return None


def train_and_evaluate(data):
    """Train and evaluate Naive Bayes model."""
    X = data['text']
    y = data['sentiment']

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Sentiment Analysis Accuracy: {accuracy:.2f}")

    return model, vectorizer


def predict_new_sample(model, vectorizer, text):
    """Predict sentiment for a new text sample."""
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    return model.predict(text_vectorized)[0]


def main():
    file_path = 'Sentiment dataset.csv'
    data = load_and_preprocess_data(file_path)
    if data is None:
        return

    model, vectorizer = train_and_evaluate(data)

    sample_text = "This product is amazing!"
    predicted_sentiment = predict_new_sample(model, vectorizer, sample_text)
    print(f"Predicted sentiment for '{sample_text}': {predicted_sentiment}")


if __name__ == "__main__":
    main()