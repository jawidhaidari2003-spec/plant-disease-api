from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os

app = FastAPI(title="Plant Disease API")

# تنظیمات CORS برای اتصال اپلیکیشن موبایل به سرور
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ۱. لود کردن اسم بیماری‌ها به صورت ایمن
try:
    with open('labels.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
except Exception as e:
    print("ارور در خواندن labels.txt:", e)
    class_names = []

# ۲. لود کردن اطلاعات بیماری‌ها از فایل جیسون
try:
    if os.path.exists('disease_info.json'):
        with open('disease_info.json', 'r', encoding='utf-8') as f:
            disease_info = json.load(f)
    else:
        disease_info = {}
except:
    disease_info = {}

# ۳. لود کردن مدل با تنسورفلو جدید (حل مشکل ورژن ۱۲)
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def home():
    return {"message": "داداش، سرور تشخیص بیماری گیاهان با قدرت روشنه!"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    try:
        # دریافت عکس از کاربر
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # تغییر سایز به ۱۲۸ در ۱۲۸ (دقیقا مثل زمان آموزش)
        image = image.resize((128, 128))
        
        # پیش‌پردازش حیاتی برای MobileNetV2
        img_array = np.array(image, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0  # همون فرمول طلایی که خروجی رو درست میکنه
        img_array = np.expand_dims(img_array, axis=0) 

        # فرستادن عکس به داخل مدل
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0] 

        # پیدا کردن کلاسی که مدل بیشترین اطمینان رو بهش داره
        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index]) * 100
        
        # پیدا کردن اسم بیماری از روی عدد
        if len(class_names) > predicted_class_index:
            predicted_class_name = class_names[predicted_class_index]
        else:
            predicted_class_name = f"Class_{predicted_class_index}"

        # استخراج اطلاعات اون بیماری از فایل جیسون
        info = disease_info.get(predicted_class_name, {"description": "اطلاعاتی یافت نشد."}) 

        # ارسال جواب نهایی
        return {
            "status": "success",
            "disease_name": predicted_class_name,
            "confidence": round(confidence, 2),
            "details": info
        } 
    except Exception as e:
        return {"status": "error", "message": f"داداش یه مشکلی پیش اومد: {str(e)}"}
