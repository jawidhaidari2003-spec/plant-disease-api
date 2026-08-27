from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
# ۱. اضافه شدن این کتابخانه برای حل مشکل CORS
from fastapi.middleware.cors import CORSMiddleware 
import numpy as np
from io import BytesIO
from PIL import Image
import ai_edge_litert.interpreter as litert

app = FastAPI()

# ۲. پیکربندی CORS: این بخش را دقیقاً زیر تعریف app قرار دادم تا قفل دسترسی مرورگر باز شود
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # به تمام دامنه‌ها از جمله لوکال‌هاست اکسپو اجازه دسترسی می‌دهد
    allow_credentials=True,
    allow_methods=["*"],  # اجازه متدهای POST, GET, OPTIONS و...
    allow_headers=["*"],  # اجازه ارسال تمام هدرها
)

# ترتیب کلاس‌ها باید دقیقاً مطابق ترتیب dataset.class_names باشد
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# بارگذاری مدل در حالت کاملاً استاندارد بدون هیچ دستکاری ابعاد
interpreter = litert.Interpreter(
    model_path="model.tflite"
)
interpreter.allocate_tensors()

# اطلاعات ورودی و خروجی مدل
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# استخراج ابعاد قفل شده مدل (مثلاً 32, 256, 256, 3)
input_shape = input_details[0]["shape"]
model_batch_size = int(input_shape[0])  # این عدد 32 است
target_height = int(input_shape[1])
target_width = int(input_shape[2])
input_dtype = input_details[0]["dtype"]


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

    # ریسایز تصویر به ابعاد مدل
    image = image.resize(
        (target_width, target_height)
    )

    img_array = np.array(
        image,
        dtype=np.float32
    )

    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        if file.content_type and not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"error": "Please upload an image file"}
            )

        file_data = await file.read()
        image = read_file_as_image(file_data)

        # ساخت یک آرایه خالی با بچ‌سایز دقیق مدل (مثلاً 32 عکس)
        full_batch = np.zeros(input_shape, dtype=np.float32)
        
        # قرار دادن عکس کاربر در اولین خانه از 32 خانه
        full_batch[0] = image

        # تبدیل نوع داده به نوع مورد نیاز مدل
        full_batch = full_batch.astype(input_dtype)

        # اجرای مدل روی بچ کامل
        interpreter.set_tensor(
            input_details[0]["index"],
            full_batch
        )

        interpreter.invoke()

        # گرفتن خروجی پیش‌بینی‌ها
        predictions = interpreter.get_tensor(
            output_details[0]["index"]
        )

        # ما فقط به نتیجه عکس اول (خانه 0) نیاز داریم
        first_image_predictions = predictions[0]

        # پیدا کردن کلاس و درصد اطمینان
        predicted_index = int(np.argmax(first_image_predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = round(float(np.max(first_image_predictions)) * 100, 2)

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
