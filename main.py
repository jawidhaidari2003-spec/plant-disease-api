from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO
from PIL import Image
import ai_edge_litert.interpreter as litert

app = FastAPI()

# ترتیب کلاس‌ها باید دقیقاً مطابق ترتیب dataset.class_names باشد
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# بارگذاری مدل
interpreter = litert.Interpreter(
    model_path="model.tflite"
)

# اطلاعات ورودی و خروجی اولیه
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# اندازه ورودی اصلی مدل (مثلاً)
input_shape = input_details[0]["shape"]

target_height = int(input_shape[1])
target_width = int(input_shape[2])
input_dtype = input_details[0]["dtype"]

# تغییر بچ‌سایز به 1 به جای 32
# ورودی جدید می‌شود: [1, target_height, target_width, 3]
new_input_shape = [1, target_height, target_width, input_shape[3]]
interpreter.resize_tensor_input(input_details[0]["index"], new_input_shape)

# حتماً بعد از تغییر سایز باید متد زیر دوباره صدا زده شود
interpreter.allocate_tensors()

# گرفتن اطلاعات جدید ورودی پس از آپدیت سایز
input_details = interpreter.get_input_details()


@app.get("/")
async def home():
    return {
        "message": "Plant Disease API is running"
    }

@app.get("/ping")
async def ping():
    return "Hello I am Rohullah"

def read_file_as_image(data):
    image = Image.open(
        BytesIO(data)
    ).convert("RGB")

    # Resize به اندازه مورد نیاز مدل
    image = image.resize(
        (target_width, target_height)
    )

    # تبدیل به NumPy
    # تقسیم بر 255 نمی‌کنیم
    # چون Rescaling داخل خود مدل وجود دارد
    img_array = np.array(
        image,
        dtype=np.float32
    )

    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # بررسی اینکه فایل واقعاً عکس باشد
        if file.content_type and not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Please upload an image file"
                }
            )

        # خواندن عکس
        file_data = await file.read()

        # آماده‌سازی عکس
        image = read_file_as_image(file_data)

        # اضافه کردن Batch Dimension
        img_batch = np.expand_dims(
            image,
            axis=0
        )

        # تبدیل dtype در صورت نیاز
        img_batch = img_batch.astype(input_dtype)

        # اجرای مدل
        interpreter.set_tensor(
            input_details[0]["index"],
            img_batch
        )

        interpreter.invoke()

        # گرفتن پیش‌بینی
        predictions = interpreter.get_tensor(
            output_details[0]["index"]
        )

        # گرفتن کلاس
        predicted_index = int(
            np.argmax(predictions[0])
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        # Confidence
        confidence = round(
            float(np.max(predictions[0])) * 100,
            2
        )

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:

        print(f"PREDICTION ERROR: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
