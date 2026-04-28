import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()
    
    # remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove "subject" prefix
    text = re.sub(r'^subject\s*:\s*', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove extra spaces
    text = text.strip()

    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


def preprocess_series(X):
    return X.apply(clean_text)