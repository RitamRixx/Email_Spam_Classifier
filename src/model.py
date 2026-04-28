from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import os


def build_model():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2)
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    return pipeline


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)

    os.makedirs("reports", exist_ok=True)
    
    with open("reports/classification_report.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)