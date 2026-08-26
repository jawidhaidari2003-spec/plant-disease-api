from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
import os  # برای خواندن پورت سرور اضافه شد

# آدرس ویندوز حذف شد و آدرس نسبی فایل مدل که در گیت‌هاب قرار دارد جایگزین شد
MODEL = tf.keras.models.load_model("3.keras")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello I am Rohullah"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    pred = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(pred[0])]
    confidence = np.max(pred[0])  
    
    return {
        "Class": predicted_class, 
        "Confidence": float(confidence)
    }

if __name__ == "__main__":
    # تنظیم پورت به صورت پویا برای اینکه رندر بتونه پورت سرور رو مدیریت کنه
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
