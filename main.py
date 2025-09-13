from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import pipeline
import base64

app = FastAPI()

# مدل رایگان متن به متن
story_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

@app.post("/generate_storyboard/")
async def generate_storyboard(text: str = Form(...), image: UploadFile = None):
    try:
        # اگر تصویر آپلود شده، فقط اطلاع میدیم (Base64 optional)
        image_info = ""
        if image:
            image_bytes = await image.read()
            image_info = " یک تصویر دریافت شده است."

        prompt = f"""
        شما یک کارگردان حرفه‌ای هستید.
        بر اساس متن زیر و تصویر ارسال شده (اگر موجود باشد):
        {text}{image_info}

        یک سناریو فیلم‌برداری کوتاه بسازید و استوری‌بورد مرحله‌ای بدهید:
        - برای هر پلان حرکت دوربین (Zoom, Pan, Tilt, Tracking) را توضیح دهید.
        - به زبان فارسی بنویسید.
        """

        result = story_pipeline(prompt, max_length=1024, do_sample=True)[0]["generated_text"]

        return {"storyboard": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
