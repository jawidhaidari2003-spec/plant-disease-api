from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import numpy as np
from PIL import Image
import io
import os
import urllib.request
import tflite_runtime.interpreter as tflite

app = FastAPI(title="Plant Disease API")

# تنظیمات CORS برای دسترسی فرانت‌اند و اپلیکیشن موبایل
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ۱. خواندن فایل‌های متنی و تنظیمات کلاس‌ها
with open('labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]

with open('disease_info.json', 'r', encoding='utf-8') as f:
    disease_info = json.load(f)

MODEL_PATH = "plant_disease_model.tflite"

# 🟢 ۲. لینک مستقیم دانلود فایل ۸ مگابایتی واقعی شما برای دور زدن مشکل Git LFS
# لینک فایل آپلود شده خودت رو جایگزین این لینک نمونه کن
DOWNLOAD_URL = "https://dropbox.com"

def ensure_model_exists():
    """این تابع بررسی میکنه که اگر فایل مدل خراب یا پوینتر متنی بود، نسخه واقعی رو دانلود کنه"""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100 * 1024:
        print("🔄 Model file is missing or corrupt (Git LFS issue). Downloading real 8MB model...")
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            
            # دانلود مستقیم فایل با بایت‌های کامل
            urllib.request.urlretrieve(DOWNLOAD_URL, MODEL_PATH)
            print(f"✅ Download complete! File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"❌ Failed to download model: {str(e)}")

# اجرای فرآیند چک کردن سلامت فایل قبل از لود شدن برنامه
ensure_model_exists()

# ۳. راه‌اندازی اینترپرتر مدل تاف‌لایت به صورت داینامیک و امن
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ SUCCESS: TFLite Interpreter allocated successfully on Render!")
except Exception as e:
    print(f"❌ CRITICAL ERROR DURING INTERPRETER ALLOCATION: {str(e)}")
    raise e

@app.get("/")
def home():
    return {"message": "داداش، سرور تشخیص بیماری گیاهان با قدرت روشنه!"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        # ۴. خواندن و آماده‌سازی عکس ورودی
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((128, 128))
        
        # تبدیل به آرایه اعشاری
        img_array = np.array(image, dtype=np.float32)
        
        # اعمال دقیق فرمول نرمالایزیشن موبایل‌نت روی پیکسل‌ها (بین ۱- و ۱)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0) 

        # ۵. اجرای استنتاج روی مدل TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0] 

        # پیدا کردن بهترین کلاس و محاسبه درصد اطمینان
        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index]) * 100
        predicted_class_name = class_names[predicted_class_index] 

        # گرفتن اطلاعات فارسی بیماری از فایل JSON
        info = disease_info.get(predicted_class_name, {"description": "اطلاعاتی برای این بیماری یافت نشد."}) 

        return {
            "status": "success",
            "disease_name": predicted_class_name,
            "confidence": round(confidence, 2),
            "details": info
        } 
    except Exception as e:
        return {"status": "error", "message": f"داداش یه مشکلی پیش اومد: {str(e)}"}
