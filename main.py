from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from PIL import Image
import io
import onnxruntime as ort

app = FastAPI(title="Plant Disease ONNX API")

# تنظیمات CORS برای دسترسی اپلیکیشن موبایل و وب بدون محدودیت
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ۱. خواندن فایل‌های متنی کلاس‌ها و دیتابیس توضیحات فارسی
with open('labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

with open('disease_info.json', 'r', encoding='utf-8') as f:
    disease_info = json.load(f)

# ۲. لود کردن مدل انیکس
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
        # ۳. دریافت و آماده‌سازی تصویر
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((128, 128))
        
        img_array = np.array(image, dtype=np.float32)
        
        # نرمالایزیشن استاندارد موبایل‌نت کدهای آموزش شما (بردن پیکسل‌ها بین ۱- و ۱)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0) 

        # ۴. اجرای استنتاج روی مدل انیکس
        raw_preds = session.run(None, {input_name: img_array})

        # 🟢 اصلاح حیاتی و طلایی: استخراج لایه اول خروجی آرایه برای شکستن قفل خروجی Unknown
        # مدل‌های خروجی کراس به انیکس، آرایه احتمالات را به صورت یک لیست سه بعدی یا دو بعدی برمی‌گردانند
        predictions = np.squeeze(raw_preds[0])

        # پیدا کردن بهترین کلاس واقعی خروجی
        predicted_class_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class_index])
        
        # تبدیل خودکار به درصد اگر فرمت خروجی اعشاری زیر ۱ بود
        if confidence <= 1.0:
            confidence = confidence * 100
            
        predicted_class_name = class_names[predicted_class_index] 

        # گرفتن جزئیات فارسی از فایل JSON شما
        info = disease_info.get(predicted_class_name, {
            "نام بیماری": predicted_class_name,
            "عامل بیماری": "مشخص نشده",
            "علائم معمول": "توضیحی ثبت نشده است."
        }) 

        return {
            "status": "success",
            "disease_name": predicted_class_name,
            "confidence": round(confidence, 2),
            "details": info
        } 
    except Exception as e:
        return {"status": "error", "message": f"داداش مشکلی پیش آمد: {str(e)}"}
