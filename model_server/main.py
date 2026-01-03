from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import io
import torch

app = FastAPI()

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("./my_final_model.pt")
model.to(device)

def validate_image(file: UploadFile):
    # 1. Check MIME type (Basic filter)
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG or PNG allowed.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # --- VALIDATION ---
    validate_image(file)
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 2. Check for truncated/corrupt images
        image.verify() 
        # Re-open after verify (verify closes the file pointer)
        image = Image.open(io.BytesIO(contents))
        
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not identify image file.")

    # --- PREPROCESSING ---
    # Convert to RGB to handle PNGs with transparency or Grayscale images
    if image.mode != "RGB":
        image = image.convert("RGB")

    # --- INFERENCE ---
    # conf=0.5: Only return detections with >50% confidence
    # iou=0.5:  Non-Maximum Suppression threshold (removes overlapping boxes)
    results = model(image, conf=0.5, iou=0.5)

    # --- POST-PROCESSING ---
    detections = []
    
    # YOLOv8 returns a list of result objects (we only sent 1 image, so take index 0)
    result = results[0]
    
    for box in result.boxes:
        # Extract data
        x1, y1, x2, y2 = box.xyxy[0].tolist() # Bounding box coordinates
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        
        detections.append({
            "label": class_name,
            "confidence": round(confidence, 2),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

    return {
        "filename": file.filename,
        "count": len(detections),
        "detections": detections
    }