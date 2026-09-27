from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from PIL import Image
import io
import onnxruntime as ort

app = FastAPI(title="Plant Disease ONNX API")

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

MODEL_PATH = "plant_model.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print("✅ SUCCESS: ONNX Model loaded perfectly on Render!")
except Exception as e:
    print(f"❌ ERROR LOADING ONNX MODEL: {str(e)}")
    raise e

@app.get("/")
def home():
    return {"message": "داداش، سرور هوشمند گیاه‌پزشکی با قدرت روشنه!"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((128, 128))
        
        # تبدیل به آرایه شناور استاندارد
        img_array = np.array(image, dtype=np.float32)
        
        # 🟢 اصلاح طلایی: هماهنگ‌سازی دقیق با متد پیش‌پردازش کدهای آموزش شما
        # پیکسلهای تصویر را دقیقاً به بازه [1-, 1] می‌بریم
        img_array = (img_array / 127.5) - 1.0
        
        # اضافه کردن بعد بچ (Batch Dimension) -> (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0) 

        # اجرای استنتاج روی مدل انیکس
        raw_preds = session.run(None, {input_name: img_array})[0][0]

        # پیدا کردن کلاسی که بیشترین امتیاز رو آورده
        predicted_class_index = int(np.argmax(raw_preds))
        confidence = float(raw_preds[predicted_class_index])
        
        # اگر خروجی مدل به صورت درصد مستقیم نبود، ضربدر ۱۰۰ میکنیم
        if confidence <= 1.0:
            confidence = confidence * 100
            
        predicted_class_name = class_names[predicted_class_index] 

        # استخراج اطلاعات فارسی بیماری از دیتابیس جی‌سان شما
        info = disease_info.get(predicted_class_name, {
            "نام بیماری": predicted_class_name,
            "description": "اطلاعات تکمیلی برای این کلاس یافت نشد."
        }) 

        return {
            "status": "success",
            "disease_name": predicted_class_name,
            "confidence": round(confidence, 2),
            "details": info
        } 
    except Exception as e:
        return {"status": "error", "message": f"داداش مشکلی پیش آمد: {str(e)}"}
