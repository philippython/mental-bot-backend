from fastapi import FastAPI
from pydantic import BaseModel
from model.chat_bot import classify_statement
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = ["*"]

# cors middleware setting
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class Message(BaseModel):
    text: str

@app.post("/predict")
def predict_emotion(msg: Message):
    prediction = classify_statement(msg.text)
    return {
        "prediction": prediction,
    }
