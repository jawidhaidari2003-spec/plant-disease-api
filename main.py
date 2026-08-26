from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image

# لود کردن مدل فوق‌سبک TFLite (این دستور رم سرور را پر نمی‌کند)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello I am Rohullah"

def read_file_as_image(data) -> np.ndarray:
    # تغییر اندازه عکس به سایز استاندارد ۲۵۶ در ۲۵۶ که برای مدل‌های سی‌ان‌ان رایج است
    image = Image.open(BytesIO(data)).resize((256, 256))
    img_array = np.array(image, dtype=np.float32)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    # اجرای مدل لایت
    interpreter.set_tensor(input_details['index'], img_batch)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details['index'])
    
    predicted_class = CLASS_NAMES[np.argmax(pred)]
    confidence = np.max(pred)  
    
    return {
        "Class": predicted_class, 
        "Confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
