from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io
import numpy as np
import cv2

from ultralytics import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup (IMPORTANT for Render)
MODEL_PATH = "best.pt"   # make sure this exists in your repo / container
model = YOLO(MODEL_PATH)

@app.get("/")
def health():
    return {"status": "ok"}

def _get_upload(file: Optional[UploadFile], image: Optional[UploadFile]):
    return file or image

def _read_image_to_bgr(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img

@app.post("/predict_image")
async def predict_image(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    upload = _get_upload(file, image)
    if upload is None:
        ct = request.headers.get("content-type", "")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "No image received.",
                "expected": "multipart/form-data with field 'file' or 'image'",
                "received_content_type": ct,
            },
        )

    img_bgr = _read_image_to_bgr(upload)

    # Run YOLO segmentation
    results = model.predict(img_bgr, conf=0.25)
    r = results[0]

    # Ultralytics can render an overlay image for you:
    plotted = r.plot()  # returns BGR image with masks/boxes/labels drawn

    # Encode to PNG and return
    ok, buf = cv2.imencode(".png", plotted)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
