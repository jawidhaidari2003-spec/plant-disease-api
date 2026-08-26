from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import ai_edge_litert.interpreter as litert 

# بارگذاری مدل سبک
interpreter = litert.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# استخراج دقیق طول و عرض مورد نیاز مدل از روی ساختار تنسور ورودی
# برای مثال اگر شکل ورودی [1, 256, 256, 3] باشد، مقادیر 256 استخراج می‌شوند
_, target_height, target_width, _ = input_details[0]['shape']

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello I am Rohullah"

def read_file_as_image(data) -> np.ndarray:
    # ۱. تبدیل عکس به ۳ کانال رنگی استاندارد (RGB)
    image = Image.open(BytesIO(data)).convert("RGB")
    
    # ۲. تغییر سایز دقیق عکس بر اساس طول و عرض مورد نیاز مدل تو
    image = image.resize((target_width, target_height))
    
    # ۳. تبدیل به آرایه اعشاری و نرمال‌سازی پیکسل‌ها بین ۰ و ۱
    img_array = np.array(image, dtype=np.float32) / 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    
    # اضافه کردن بعد Batch به آرایه عکس
    img_batch = np.expand_dims(image, axis=0)
    
    # اجرای سریع مدل لایت
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = CLASS_NAMES[np.argmax(pred[0])]
    confidence = np.max(pred[0])  
    
    return {
        "Class": predicted_class, 
        "Confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)
