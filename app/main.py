from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.model import predict_image

app = FastAPI(title="YOLOv11-Seg API")

# ✅ CORS (for FlutterFlow Web / browser preflight)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or set your FlutterFlow domain here
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_image(contents)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
