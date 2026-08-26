from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
# استفاده از کتابخانه فوق سبک جدید گوگل به جای تنسورفلو سنگین
import ai_edge_litert as litert 

# لود کردن مدل با ابزار سبک لایت‌آرتی
interpreter = litert.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello I am Rohullah"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256, 256))
    img_array = np.array(image, dtype=np.float32)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = CLASS_NAMES[np.argmax(pred)]
    confidence = np.max(pred)  
    
    return {
        "Class": predicted_class, 
        "Confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)
