# main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

from utils.overlay import annotate_image
from utils.croqui import draw_scene_croqui
from utils.pdf_export import export_storyboard_pdf

app = FastAPI()
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# زاویه‌های ممکن دوربین
angles = ["Eye level", "High", "Low", "Overhead"]

# بارگذاری مدل YOLOv8
yolo_model = YOLO("yolov8n.pt")  # دفعه اول مدل دانلود می‌شود

# تحلیل اشیاء تصویر
def analyze_scene_objects(image_path: str):
    results = yolo_model.predict(source=image_path, verbose=False)
    objects = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = yolo_model.names[int(cls)]
            objects.append({"type": label, "bbox": [x1, y1, x2, y2]})
    return objects

# تحلیل هوشمند شات
def generate_shot_smart(objects, image_shape, idx):
    h, w = image_shape[:2]
    bboxes = np.array([obj['bbox'] for obj in objects]) if objects else np.array([[0,0,w,h]])
    centers = (bboxes[:,:2] + bboxes[:,2:])/2
    area_ratio = np.sum((bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1]))/(w*h) if objects else 0
    center = np.mean(centers, axis=0)

    angle_score = center[1]/h
    angle_index = int(np.clip(angle_score*3, 0, 3))
    selected_angle = angles[angle_index]

    movement = "Static" if len(bboxes)<2 or area_ratio>0.3 else ("Pan" if np.std(centers[:,0])>w*0.2 else "Tilt")
    duration = max(3, int(len(bboxes) + np.std(centers[:,1])/50))

    description = f"Scene with {len(bboxes)} objects. Camera {selected_angle} with {movement}. Duration: {duration} sec."
    return {
        "id": idx+1,
        "title": f"Plan {idx+1}",
        "angle": selected_angle,
        "movement": movement,
        "duration_sec": duration,
        "description": description
    }

# API اصلی
@app.post("/generate-storyboard")
async def generate_storyboard(files: list[UploadFile], prompt: str = Form(...)):
    try:
        session_id = str(uuid.uuid4())
        session_path = os.path.join(OUTPUT_DIR, session_id)
        os.makedirs(session_path, exist_ok=True)

        shots, annotated_images, croqui_images = [], [], []

        for idx, file in enumerate(files):
            file_path = os.path.join(session_path, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # تحلیل صحنه
            objects = analyze_scene_objects(file_path)
            image = cv2.imread(file_path)
            shot_data = generate_shot_smart(objects, image.shape, idx)

            # Annotated image
            annotated_path = os.path.join(session_path, f"annotated_{idx+1}.png")
            annotate_image(file_path, f"{shot_data['title']}: {shot_data['description']}", annotated_path)

            # Croqui image
            croqui_path = os.path.join(session_path, f"croqui_{idx+1}.png")
            draw_scene_croqui(objects, croqui_path)

            shots.append(shot_data)
            annotated_images.append(annotated_path)
            croqui_images.append(croqui_path)

        # PDF استوری‌بورد
        pdf_path = os.path.join(session_path, "storyboard.pdf")
        export_storyboard_pdf(shots, annotated_images, croqui_images, pdf_path)

        return JSONResponse({
            "prompt": prompt,
            "shots": shots,
            "annotated_images": annotated_images,
            "croqui_images": croqui_images,
            "storyboard_pdf": pdf_path
        })

    except Exception as e:
        # نمایش خطا در JSON برای دیباگ
        return JSONResponse({"error": str(e)})
