from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import json
import os

app = FastAPI()

# ==================== CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== مسیرها ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.tflite")
DATA_PATH = os.path.join(BASE_DIR, "disease_info.json")

# ==================== چک کردن فایل‌ها ====================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Disease data file not found: {DATA_PATH}")

# ==================== لود فایل JSON ====================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    disease_info = json.load(f)

CLASS_NAMES = list(disease_info.keys())
n_classes = len(CLASS_NAMES)
print(f"Loaded {n_classes} classes from disease_info.json")

if n_classes != 63:
    raise ValueError(f"Expected 63 classes, but found {n_classes} classes in disease_info.json")

# ==================== لود مدل TFLite ====================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
target_height = int(input_shape[1])
target_width = int(input_shape[2])

print(f"Model input shape: {input_shape}")
print(f"Model input dtype: {input_dtype}")
print(f"Model output shape: {output_details[0]['shape']}")

# ==================== توابع کمکی ====================
def read_file_as_image(data):
    """
    عکس رو می‌خونه، به سایز مدل تغییر میده و نرمال‌سازی می‌کنه (تقسیم بر ۲۵۵).
    این دقیقاً همون کاریه که توی Colab انجام شد.
    """
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((target_width, target_height))
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0   # 👈 نرمال‌سازی
    return image_array

def prepare_input(image):
    model_input = np.expand_dims(image, axis=0)
    return model_input.astype(input_dtype)

# ==================== Endpoints ====================
@app.get("/")
async def home():
    return {"message": "Plant Disease API is running"}

@app.get("/ping")
async def ping():
    return {"message": "Hello I am Rohullah"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # چک کردن نوع فایل
        if file.content_type and not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload an image file"}
            )

        file_data = await file.read()
        if not file_data:
            return JSONResponse(
                status_code=400,
                content={"error": "The uploaded image is empty"}
            )

        # پردازش عکس
        image = read_file_as_image(file_data)
        model_input = prepare_input(image)

        # پیش‌بینی
        interpreter.set_tensor(input_details[0]["index"], model_input)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])
        first_image_predictions = predictions[0]

        # پیدا کردن کلاس با بالاترین احتمال
        predicted_index = int(np.argmax(first_image_predictions))
        if predicted_index >= len(CLASS_NAMES):
            raise ValueError("Model output classes do not match disease_info.json classes.")

        predicted_class = CLASS_NAMES[predicted_index]
        raw_confidence = float(np.max(first_image_predictions))

        if raw_confidence <= 1.0:
            confidence = raw_confidence * 100
        else:
            confidence = raw_confidence

        confidence = round(confidence, 2)

        # اگه اطمینان کم بود
        if confidence < 60:
            return {
                "status": "low_confidence",
                "message": "لطفا یک عکس واضح از برگ گیاه یا قسمت آسیب دیده برگ ارسال کنید.",
                "confidence": confidence
            }

        # گرفتن اطلاعات بیماری
        info = disease_info.get(predicted_class, {})

        # اگه کلاس Unknown بود
        if predicted_class == "Unknown":
            return {
                "status": "unknown",
                "message": "این تصویر شبیه برگ گیاه نیست. لطفاً یک عکس واضح از برگ گیاه ارسال کنید.",
                "confidence": confidence
            }

        return {
            "status": "success",
            "class": predicted_class,
            "confidence": confidence,
            "info": info
        }

    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "detail": str(e)}
        )
