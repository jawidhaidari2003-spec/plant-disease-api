from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from keras import models
from tensorflow import keras

MODEL: keras.Model = keras.models.load_model(r"E:\Excersize\models\3.keras")


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI()

@app.get("/ping")

async def ping():
    return "Hello I am Rohullah"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")

async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image( await file.read())
    
    img_batch = np.expand_dims(image, 0)
    
    pred = MODEL.predict(img_batch)
    
    predicred_class = CLASS_NAMES[np.argmax(pred[0])]
    
    confidence = np.max(pred[0])  
    
    return {
        "Class": predicred_class, 
        "Confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)    