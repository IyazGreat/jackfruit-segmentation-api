from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import shutil
import uuid

app = FastAPI()

# If FlutterFlow calls from web (or you embed in web), CORS helps.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok", "message": "Service is running. Use POST /predict with multipart file."}

# Optional: avoid confusion when someone opens /predict in browser
@app.get("/predict")
def predict_get_help():
    return {
        "message": "Use POST /predict with multipart/form-data.",
        "accepted_fields": ["file", "image"],
        "example": "POST with form-data: file=<your_image>"
    }

@app.post("/predict")
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Accepts multipart/form-data with either:
      - file=<image>
      - image=<image>
    Works for Postman + FlutterFlow even if you used the wrong key name.
    """

    # Pick whichever was provided
    upload = file or image
    if upload is None:
        # Extra debug: show what content-type was received
        ct = request.headers.get("content-type", "")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "No image file received.",
                "expected": "multipart/form-data with a file field named 'file' or 'image'",
                "received_content_type": ct,
                "flutterflow_tip": "Set API Call -> POST -> Multipart Form-Data, add field key 'file' (type File). Don't set Content-Type manually."
            }
        )

    # Basic content type check (optional but nice)
    if upload.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {upload.content_type}. Use JPG/PNG/WebP."
        )

    # Save to temp
    temp_dir = "tmp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    ext = os.path.splitext(upload.filename or "")[-1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

    # =========================
    # TODO: Run your YOLO here
    # =========================
    # Example placeholder result:
    # result = yolo_predict(temp_path)

    # Clean up (optional)
    # os.remove(temp_path)

    return {
        "ok": True,
        "received_filename": upload.filename,
        "received_as": "file" if file else "image",
        "content_type": upload.content_type,
        "note": "Replace placeholder with your YOLO prediction output."
    }
