# main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
import json

from utils.overlay import annotate_image
from utils.croqui import draw_scene_croqui
from utils.pdf_export import export_storyboard_pdf

app = FastAPI()
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

angles = ["Eye level", "High", "Low", "Overhead"]

yolo_model = YOLO("yolov8n.pt")

# مدل NLP برای سناریو کامل سینماتیک با زمان‌بندی و Transition
storyboard_model = pipeline(
    "text2text-generation",
    model="HooshvareLab/bert-fa-base-uncased"  # یا مدل fine-tuned برای تولید سناریو
)

def analyze_scene_objects(image_path: str):
    results = yolo_model.predict(source=image_path, verbose=False)
    objects = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = yolo_model.names[int(cls)]
            objects.append({"type": label, "bbox": [x1, y1, x2, y2]})
    return objects

# تولید سناریو کامل با دیالوگ، زمان‌بندی و Transition
def generate_cinematic_storyboard(prompt_text: str):
    """
    مدل NLP متن را تحلیل می‌کند و JSON شامل سبک و شات‌های سینماتیک تولید می‌کند.
    هر شات شامل زاویه، حرکت، مدت زمان، تمرکز، توضیح تصویری، دیالوگ و Transition است.
    """
    try:
        output = storyboard_model(prompt_text, max_length=1024)
        generated_text = output[0]['generated_text']
        analysis = json.loads(generated_text)
        # اگر طول شات‌ها جمعا کمتر یا بیشتر از 60 ثانیه بود، بازتوزیع زمان
        total_duration = sum([s.get("duration_sec",3) for s in analysis.get("shots", [])])
        if total_duration > 0:
            factor = 60 / total_duration
            for s in analysis.get("shots", []):
                s["duration_sec"] = max(1, int(s.get("duration_sec",3)*factor))
        return analysis
    except Exception as e:
        return {"style": "سینماتیک", "shots": []}

def generate_shot_with_image(objects, image_shape, shot_info, idx):
    h, w = image_shape[:2]
    bboxes = np.array([obj['bbox'] for obj in objects]) if objects else np.array([[0,0,w,h]])
    centers = (bboxes[:,:2] + bboxes[:,2:])/2
    center = np.mean(centers, axis=0)

    selected_angle = shot_info.get("angle", angles[int(np.clip(center[1]/h*3,0,3))])
    movement = shot_info.get("movement", "Static")
    duration = shot_info.get("duration_sec", max(3, int(len(bboxes) + np.std(centers[:,1])/50)))
    description = shot_info.get("description", "")
    dialogue = shot_info.get("dialogue", "")
    transition = shot_info.get("transition", "Cut")

    return {
        "id": idx+1,
        "title": f"Shot {idx+1}",
        "angle": selected_angle,
        "movement": movement,
        "duration_sec": duration,
        "description": description,
        "dialogue": dialogue,
        "transition": transition
    }

@app.post("/generate-storyboard")
async def generate_storyboard(files: list[UploadFile], prompt: str = Form(...)):
    try:
        session_id = str(uuid.uuid4())
        session_path = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(session_path, exist_ok=True)

        shots, annotated_images, croqui_images = [], [], []

        cinematic_storyboard = generate_cinematic_storyboard(prompt)

        for idx, file in enumerate(files):
            file_path = os.path.join(session_path, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            objects = analyze_scene_objects(file_path)
            image = cv2.imread(file_path)

            shot_info = cinematic_storyboard.get("shots", [{}])[idx] if idx < len(cinematic_storyboard.get("shots", [])) else {}

            shot_data = generate_shot_with_image(objects, image.shape, shot_info, idx)

            annotated_path = os.path.join(session_path, f"annotated_{idx+1}.png")
            annotate_image(
                file_path,
                f"{shot_data['title']}:\n{shot_data['description']}\nDialog: {shot_data['dialogue']}\nTransition: {shot_data['transition']}",
                annotated_path
            )

            croqui_path = os.path.join(session_path, f"croqui_{idx+1}.png")
            draw_scene_croqui(objects, croqui_path)

            shots.append(shot_data)
            annotated_images.append(annotated_path)
            croqui_images.append(croqui_path)

        pdf_path = os.path.join(session_path, "storyboard.pdf")
        export_storyboard_pdf(shots, annotated_images, croqui_images, pdf_path)

        return JSONResponse({
            "prompt": prompt,
            "cinematic_storyboard": cinematic_storyboard,
            "shots": shots,
            "annotated_images": annotated_images,
            "croqui_images": croqui_images,
            "storyboard_pdf": pdf_path
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})
