import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
import ai_edge_litert.interpreter as litert
from PIL import Image
import io

app = FastAPI()

# استفاده از مفسر جدید گوگل برای حل ارور FULLY_CONNECTED
interpreter = litert.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("disease_info.json", "r", encoding="utf-8") as f:
    disease_info = json.load(f)

CLASS_NAMES = [
"Apple___Apple_scab",
"Apple___Black_rot",
"Apple___Cedar_apple_rust",
"Apple___healthy",
"Banana_cordana",
"Banana_healthy",
"Banana_pestalotiopsis",
"Banana_sigatoka",
"Beans_angular_leaf_spot",
"Beans_bean_rust",
"Beans_healthy",
"Blueberry___healthy",
"Cherry_(including_sour)___Powdery_mildew",
"Cherry_(including_sour)___healthy",
"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
"Corn_(maize)___Common_rust_",
"Corn_(maize)___Northern_Leaf_Blight",
"Corn_(maize)___healthy",
"Eggplant_Healthy Leaf",
"Eggplant_Insect Pest Disease",
"Eggplant_Leaf Spot Disease",
"Eggplant_Mosaic Virus Disease",
"Eggplant_Small Leaf Disease",
"Eggplant_White Mold Disease",
"Eggplant_Wilt Disease",
"Grapes_Grape___Black_rot",
"Grapes_Grape___Esca_(Black_Measles)",
"Grapes_Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
"Grapes_Grape___healthy",
"Orange___Haunglongbing_(Citrus_greening)",
"Orange_sales_Citrus_Canker_Diseases_Leaf_Orange",
"Orange_sales_Citrus_Nutrient_Deficiency_Yellow_Leaf_Orange",
"Orange_sales_Multiple_Diseases_Leaf_Orange",
"Orange_sales_Young_Healthy_Leaf_Orange",
"Peach___Bacterial_spot",
"Peach___health",
"Pepper,_bell___Bacterial_spot",
"Pepper,_bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Pumpkin_Bacterial Leaf Spot",
"Pumpkin_Downy Mildew",
"Pumpkin_Healthy Leaf",
"Pumpkin_Mosaic Disease",
"Pumpkin_Powdery_Mildew",
"Squash___Powdery_mildew",
"Tomato___Bacterial_spot",
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Tomato_mosaic_virus",
"Tomato___healthy",
"Unknown",
"cucumber_Downy_mildew",
"cucumber_Healthy_leaves",
"cucumber_Powdery_mildew",
"stawberry_Strawberry___Leaf_scorch",
"stawberry_Strawberry___healthy"    
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    
    img_array = np.array(image, dtype=np.float32)
    single_input = np.expand_dims(img_array, axis=0)
    
    try:
        interpreter.resize_tensor_input(input_details[0]['index'], single_input.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], single_input)
    except Exception:
        batch_input = np.repeat(single_input, 8, axis=0)
        interpreter.resize_tensor_input(input_details[0]['index'], batch_input.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], batch_input)
    
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if len(output_data.shape) > 1:
        predicted_index = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0])) * 100
    else:
        predicted_index = np.argmax(output_data)
        confidence = float(np.max(output_data)) * 100
        
    predicted_class = CLASS_NAMES[predicted_index]
    extra_details = disease_info.get(predicted_class, {"info": "اطلاعاتی در فایل JSON یافت نشد."})
    
    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "details": extra_details
    }

@app.get("/")
def read_root():
    return {"status": "Server is running successfully with LiteRT!"}
