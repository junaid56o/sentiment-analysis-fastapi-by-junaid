from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

model = pickle.load(open('sentiment_model.pkl', 'rb'))
vect = pickle.load(open('vectorizer.pkl', 'rb'))

app = FastAPI()

# 🔥 ADD THIS BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}

@app.post("/predict")
def predict(text: str):
    vector = vect.transform([text])
    prediction = model.predict(vector)[0]

    prob = model.predict_proba(vector)[0]
    confidence = max(prob)

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "input": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 2)
    }