from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import base64
import numpy as np
import cv2
import time

from ultralytics import YOLO

app = FastAPI()

# Allow FlutterFlow / web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load your YOLO segmentation model
# -----------------------------
MODEL_PATH = "best.pt"   # make sure this exists in your repo / container
model = YOLO(MODEL_PATH)

@app.get("/")
def health():
    return {"status": "ok", "message": "Use POST /predict with multipart file key 'file' or 'image'."}

@app.get("/predict")
def predict_help():
    return {
        "message": "Use POST /predict with multipart/form-data.",
        "accepted_fields": ["file", "image"],
        "returns": ["detected", "class_name", "confidence", "processing_ms", "overlay_png_b64", "mask_png_b64"]
    }

def _bgr_from_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img

def _png_b64_from_bgr(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def _png_b64_from_gray(mask_gray_0_255: np.ndarray) -> str:
    if mask_gray_0_255.dtype != np.uint8:
        mask_gray_0_255 = mask_gray_0_255.astype(np.uint8)
    ok, buf = cv2.imencode(".png", mask_gray_0_255)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode mask PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

@app.post("/predict")
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    start = time.perf_counter()

    upload = file or image
    if upload is None:
        ct = request.headers.get("content-type", "")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "No image file received.",
                "expected": "multipart/form-data with field 'file' or 'image' (type File).",
                "received_content_type": ct,
                "flutterflow_tip": "Set POST + Multipart Form-Data. Add field key 'file' (Type: File). Do NOT set Content-Type manually."
            },
        )

    # Read bytes once
    image_bytes = await upload.read()
    img_bgr = _bgr_from_bytes(image_bytes)

    # Run YOLO segmentation
    results = model.predict(img_bgr, conf=0.25)
    r = results[0]

    # Collect detections
    detections: List[Dict[str, Any]] = []
    top_class_name = None
    top_conf = None

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            xyxy = [float(x) for x in b.xyxy[0].tolist()]
            name = model.names.get(cls_id, str(cls_id))
            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "confidence": conf,
                "xyxy": xyxy
            })

        best = max(detections, key=lambda d: d["confidence"])
        top_class_name = best["class_name"]
        top_conf = float(best["confidence"])

    # Overlay image (boxes + masks)
    overlay_bgr = r.plot()
    overlay_png_b64 = _png_b64_from_bgr(overlay_bgr)

    # Combined binary mask (all instances)
    mask_png_b64 = None
    if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
        masks = r.masks.data
        try:
            masks_np = masks.cpu().numpy()
        except Exception:
            masks_np = np.array(masks)

        combined = (np.max(masks_np, axis=0) > 0.5).astype(np.uint8) * 255
        mask_png_b64 = _png_b64_from_gray(combined)

    # -----------------------------
    # FINAL LABEL RULES (robust)
    # -----------------------------
    has_boxes = len(detections) > 0

    healthy_aliases = {
        "healthy",
        "healthy_jackfruit",
        "jackfruit_healthy",
        "no_disease",
        "nodisease",
        "no disease",
        "normal",
    }

    if not has_boxes:
        detected = False
        top_class_name = "Not a Jackfruit or Healthy"
        top_conf = 0.0  # or None if you prefer
    else:
        detected = True
        if isinstance(top_class_name, str) and top_class_name.strip().lower() in healthy_aliases:
            top_class_name = "Healthy Jackfruit"

    processing_ms = int((time.perf_counter() - start) * 1000)

    return JSONResponse({
        "detected": detected,
        "class_name": top_class_name,
        "confidence": top_conf,
        "processing_ms": processing_ms,
        "overlay_png_b64": overlay_png_b64,
        "mask_png_b64": mask_png_b64,
        "detections": detections,  # optional, FlutterFlow can ignore
    })
