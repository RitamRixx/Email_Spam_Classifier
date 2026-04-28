from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.explain import get_top_words

app = FastAPI(title="Spam Classifier API")


model = joblib.load("model/spam_classifier.pkl")


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Spam Classifier API is running......."}


@app.post("/predict")
def predict(request: TextRequest):
    text = request.text

    prediction = model.predict([text])[0]


    words = get_top_words(model, text)

    return {
        "prediction": int(prediction),
        "important_words": words
    }