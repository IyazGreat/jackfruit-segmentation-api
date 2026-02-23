# app/model.py
import io
import base64
import numpy as np
from PIL import Image
from ultralytics import YOLO

# MUST match your data.yaml order:
CLASS_NAMES = ["fruit borer", "fruit fly", "rot", "sclerotium"]

MODEL_PATH = "weights/best.pt"
model = YOLO(MODEL_PATH)


def _pil_to_png_base64(img: Image.Image) -> str:
    """Encode a PIL image to base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_mask_pil(mask_2d: np.ndarray) -> Image.Image:
    """
    mask_2d: HxW float/bool mask (0..1). Returns a PIL RGB image (white foreground).
    """
    mask_u8 = (mask_2d > 0.5).astype(np.uint8) * 255  # HxW uint8
    mask_img = Image.fromarray(mask_u8, mode="L").convert("RGB")
    return mask_img


def _make_overlay_pil(rgb_img: np.ndarray, mask_2d: np.ndarray) -> Image.Image:
    """
    rgb_img: HxW x 3 uint8 (RGB)
    mask_2d: HxW float/bool (0..1)
    Returns: overlay PIL image, mask colored green with alpha blending.
    """
    img = rgb_img.astype(np.float32)
    m = (mask_2d > 0.5)

    # Green overlay color in RGB
    green = np.array([0, 255, 0], dtype=np.float32)

    alpha = 0.45  # overlay strength
    img[m] = (1 - alpha) * img[m] + alpha * green

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def predict_image(image_bytes: bytes) -> dict:
    """
    Runs YOLOv11 segmentation on input image bytes.
    Returns dict for FlutterFlow API response.
    """
    # Load image as RGB
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(pil_img)  # HxWx3 uint8 (RGB)

    # Inference
    results = model.predict(source=rgb, verbose=False)
    r = results[0]

    # No detections
    if r.boxes is None or len(r.boxes) == 0:
        return {
            "detected": False,
            "class_id": None,
            "class_name": None,
            "confidence": None,
            "mask_png_b64": None,
            "overlay_png_b64": None,
        }

    # Pick best detection by confidence
    confs = r.boxes.conf.detach().cpu().numpy()
    best_i = int(np.argmax(confs))

    cls_id = int(r.boxes.cls[best_i].detach().cpu().numpy())
    conf = float(r.boxes.conf[best_i].detach().cpu().numpy())

    class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)

    mask_png_b64 = None
    overlay_png_b64 = None

    # Segmentation masks
    if r.masks is not None and r.masks.data is not None and len(r.masks.data) > best_i:
        masks = r.masks.data.detach().cpu().numpy()  # [N, H, W] float
        best_mask = masks[best_i]  # [H, W]

        mask_pil = _make_mask_pil(best_mask)
        overlay_pil = _make_overlay_pil(rgb, best_mask)

        mask_png_b64 = _pil_to_png_base64(mask_pil)
        overlay_png_b64 = _pil_to_png_base64(overlay_pil)

    return {
        "detected": True,
        "class_id": cls_id,
        "class_name": class_name,
        "confidence": conf,
        "mask_png_b64": mask_png_b64,
        "overlay_png_b64": overlay_png_b64,
    }
