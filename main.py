from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import numpy as np
from PIL import Image
import io
import tflite_runtime.interpreter as tflite

app = FastAPI(title="Plant Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

with open('disease_info.json', 'r', encoding='utf-8') as f:
    disease_info = json.load(f)

interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def home():
    return {"message": "داداش، سرور تشخیص بیماری گیاهان با قدرت روشنه!"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((128, 128))
        
        # تبدیل به آرایه اعشاری
        img_array = np.array(image, dtype=np.float32)
        
        # اصلاح حیاتی: اعمال دقیق فرمول نرمالایزیشن موبایل‌نت روی پیکسل‌ها
        img_array = (img_array / 127.5) - 1.0
        
        img_array = np.expand_dims(img_array, axis=0) 

        # اجرای مدل TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0] 

        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index]) * 100
        predicted_class_name = class_names[predicted_class_index] 

        info = disease_info.get(predicted_class_name, {"description": "اطلاعاتی برای این بیماری یافت نشد."}) 

        return {
            "status": "success",
            "disease_name": predicted_class_name,
            "confidence": round(confidence, 2),
            "details": info
        } 
    except Exception as e:
        return {"status": "error", "message": f"داداش یه مشکلی پیش اومد: {str(e)}"}
