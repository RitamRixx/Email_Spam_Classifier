import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df["text"]
    y = df["spam"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)