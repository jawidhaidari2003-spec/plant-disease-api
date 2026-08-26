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
input_shape = input_details[0]['shape'] # خواندن ابعاد دقیق مورد نیاز مدل شما

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello I am Rohullah"

def read_file_as_image(data) -> np.ndarray:
    # ۱. تبدیل عکس به ۳ کانال رنگی استاندارد برای جلوگیری از خطای تصاویر شفاف
    image = Image.open(BytesIO(data)).convert("RGB")
    
    # ۲. تغییر سایز پویا بر اساس ابعادی که مدل CNN شما به آن نیاز دارد (مثلاً 256x256)
    target_size = (input_shape[1], input_shape[2])
    image = image.resize(target_size)
    
    # ۳. تبدیل به آرایه اعشاری و تقسیم بر ۲۵۵ برای نرمال‌سازی بین ۰ و ۱
    img_array = np.array(image, dtype=np.float32) / 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    
    # اضافه کردن بعد دسته‌ای (Batch Dimension) به آرایه عکس
    img_batch = np.expand_dims(image, axis=0)
    
    # تنظیم مجدد اندازه تنسور ورودی برای همخوانی ۱۰۰ درصدی با ابعاد آرایه فرستاده شده
    interpreter.resize_tensor_input(input_details[0]['index'], img_batch.shape)
    interpreter.allocate_tensors()
    
    # اجرای مدل روی عکس بدون خطا
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
