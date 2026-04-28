from src.data_ingestion import load_data, split_data
from src.preprocessing import preprocess_series
from src.model import build_model, train_model, evaluate
import joblib
import os

def main():
    df = load_data("F:/MyProjects/PraxisProjects/Email_Spam_Classifier/data/emails.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    X_train = preprocess_series(X_train)
    X_test = preprocess_series(X_test)


    # model building
    model = build_model()
    model = train_model(model, X_train, y_train)

    joblib.dump(model, "model/spam_classifier.pkl")

    # Evaluate
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()
