# 📧 Email Spam Classifier API (NLP + FastAPI + Docker)

This project is an end-to-end NLP system that classifies email/text messages as **Spam** or **Ham (Not Spam)**.  
It includes model training, API deployment, explainability, and containerization using Docker.

---

## 🚀 Features

- TF-IDF + Random Forest based text classification
- FastAPI-based REST API (`/predict`)
- Explainability using important TF-IDF words
- Modular code structure
- Dockerized for easy deployment

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd Email_Spam_Classifier
```

---

### 2. Install dependencies

#### Using uv:
```bash
uv sync
```

#### OR using pip:
```bash
pip install -r requirements.txt
```

---

### 3. Train the model

```bash
python main.py
```

This will:
- Train the model
- Save it to: `model/spam_classifier.pkl`
- Save metrics to: `reports/metrics.json`

---

### 4. Run the API

```bash
uvicorn app:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

---

## Run with Docker

### Build Docker image
```bash
docker build -t spam-api .
```

### Run container
```bash
docker run -p 5000:5000 spam-api
```

Access API:
```
http://localhost:5000/docs
```

---

## API Usage Example

### Endpoint:
```
POST /predict
```

### Request Body:
```json
{
  "text": "Congratulations! You have won a free lottery. Click here now!"
}
```

### cURL Example:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text": "Free entry in a weekly contest! Click now!"}'
```

---

## Sample Output

```json
{
  "prediction": 1,
  "important_words": ["free", "click", "win", "offer", "now"]
}
```

Where:
- `1` → Spam  
- `0` → Ham (Not Spam)

---

## Explanation Method Used

Explainability is implemented using **TF-IDF feature importance**.

### How it works:
1. Input text is cleaned using preprocessing pipeline
2. TF-IDF assigns importance scores to each word
3. Top contributing words are extracted
4. These words are returned as explanation

### Example:

Input:
```
"You have won a free lottery"
```

Explanation:
```
["free", "won", "lottery"]
```

These words contributed most to the spam prediction.

---

## Tech Stack

- Python
- Scikit-learn
- FastAPI
- NLTK
- Docker

---

## Conclusion

This project demonstrates:
- End-to-end ML pipeline
- Model deployment via API
- Explainable AI (XAI)
- Production-ready architecture

---

## Author

Ritam Rakshit