import io
import os
import gc
import base64
import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from PIL import Image, ImageDraw
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")
IMG_SIZE   = int(os.getenv("IMG_SIZE", 224))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Focal Loss (needed to load checkpoint if saved with full model) ───────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    model = models.densenet169(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2),
    )
    return model


def load_model(path: str, device: torch.device) -> nn.Module:
    model = build_model()
    checkpoint = torch.load(path, map_location=device)

    # Support both formats: raw state_dict or dict with 'model_state_dict'
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        best_acc   = checkpoint.get("best_acc", "N/A")
        logger.info(f"Checkpoint loaded — best_acc: {best_acc}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess_2d(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load a PNG/JPG/BMP image and convert to model input tensor.
    Also returns the original resized RGB array (uint8) for visualization.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr_rgb = np.stack([arr] * 3, axis=-1)

    # Resized uint8 version for visualization (before normalization)
    vis_arr = (np.array(
        Image.fromarray((arr_rgb * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
    )).copy()

    return TRANSFORM(arr_rgb).unsqueeze(0), vis_arr


def preprocess_nifti(file_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load a NIfTI volume and extract the middle axial slice.
    Also returns the original resized RGB array (uint8) for visualization.
    """
    tmp_path = "/tmp/upload.nii"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    img_data = nib.load(tmp_path).get_fdata()
    mid       = img_data.shape[2] // 2
    sl        = img_data[:, :, mid].astype(np.float32)
    sl        = (sl - sl.min()) / (sl.max() - sl.min() + 1e-5)
    arr_rgb   = np.stack([sl] * 3, axis=-1)

    vis_arr = (np.array(
        Image.fromarray((arr_rgb * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
    )).copy()

    return TRANSFORM(arr_rgb).unsqueeze(0), vis_arr


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """Grad-CAM on the last DenseNet denseblock."""

    def __init__(self, model: nn.Module):
        self.model      = model
        self.gradients  = None
        self.activations = None
        # DenseNet-169: last dense block is features.denseblock4
        target_layer = model.features.denseblock4
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Return a heatmap (H×W float32 in [0,1])."""
        tensor = tensor.to(DEVICE)
        tensor.requires_grad_(True)

        output = self.model(tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)  # (H, W)
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def heatmap_to_bbox(cam: np.ndarray, threshold: float = 0.4) -> Optional[tuple]:
    """
    Threshold the Grad-CAM heatmap and return the bounding box
    (x_min, y_min, x_max, y_max) of the largest connected region, or None.
    """
    binary = (cam >= threshold).astype(np.uint8) * 255
    # Resize to IMG_SIZE x IMG_SIZE
    binary_resized = cv2.resize(binary, (IMG_SIZE, IMG_SIZE))
    contours, _ = cv2.findContours(binary_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Pick the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)


def draw_bbox_on_image(vis_arr: np.ndarray, bbox: tuple) -> np.ndarray:
    """Draw a red bounding box on the image and return the annotated image."""
    img_pil = Image.fromarray(vis_arr.astype(np.uint8))
    draw    = ImageDraw.Draw(img_pil)
    x0, y0, x1, y1 = bbox
    # Draw thick red rectangle
    for offset in range(3):
        draw.rectangle(
            [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
            outline=(255, 0, 0),
        )
    return np.array(img_pil)


def encode_image_base64(arr: np.ndarray) -> str:
    """Encode a numpy image array as base64 PNG string."""
    img_pil = Image.fromarray(arr.astype(np.uint8))
    buf     = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ── TTA predict ───────────────────────────────────────────────────────────────
def tta_predict(model: nn.Module, tensor: torch.Tensor, vis_arr: np.ndarray) -> dict:
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        p1 = torch.softmax(model(tensor), dim=1)
        p2 = torch.softmax(model(torch.flip(tensor, dims=[3])), dim=1)
        probs = (p1 + p2) / 2

    prob_tumor   = probs[0, 1].item()
    prob_healthy = probs[0, 0].item()
    label        = "TUMEUR" if prob_tumor > prob_healthy else "SAIN"

    result = {
        "classe":              label,
        "probabilite_tumeur":  round(prob_tumor, 4),
        "probabilite_sain":    round(prob_healthy, 4),
        "confiance":           round(max(prob_tumor, prob_healthy), 4),
    }

    # ── Visualization ─────────────────────────────────────────────────────────
    if label == "TUMEUR":
        # Generate Grad-CAM and draw bounding box
        try:
            grad_cam  = GradCAM(model)
            cam       = grad_cam.generate(tensor.clone(), class_idx=1)
            bbox      = heatmap_to_bbox(cam, threshold=0.4)

            if bbox is not None:
                annotated = draw_bbox_on_image(vis_arr.copy(), bbox)
                result["image_annotee"]   = encode_image_base64(annotated)
                result["bounding_box"]    = {
                    "x_min": bbox[0], "y_min": bbox[1],
                    "x_max": bbox[2], "y_max": bbox[3],
                }
            else:
                # No bounding box found — still return the plain image
                result["image_annotee"] = encode_image_base64(vis_arr)
                result["bounding_box"]  = None
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}. Returning plain image.")
            result["image_annotee"] = encode_image_base64(vis_arr)
            result["bounding_box"]  = None
    else:
        # SAIN — return the plain image without annotation
        result["image_annotee"] = encode_image_base64(vis_arr)
        result["bounding_box"]  = None

    return result


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Liver Tumor Detector",
    description="DenseNet-169 classifier: SAIN vs TUMEUR on CT slices",
    version="1.0.0",
)

model: Optional[nn.Module] = None


@app.on_event("startup")
def startup_event():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH, DEVICE)


@app.on_event("shutdown")
def shutdown_event():
    global model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/health", summary="Health check")
def health():
    return {
        "status":   "healthy",
        "model":    MODEL_PATH,
        "device":   str(DEVICE),
        "img_size": IMG_SIZE,
    }


@app.post("/predict", summary="Predict from 2D image (PNG/JPG/BMP)")
async def predict_2d(file: UploadFile = File(...)):
    """
    Upload a 2-D radiological image (PNG, JPG, BMP).
    Returns the classification result with probabilities and annotated image.
    - If TUMEUR: image with red bounding box around the tumor region (Grad-CAM).
    - If SAIN: plain image without annotation.
    The field `image_annotee` contains a base64-encoded PNG (data URI).
    """
    allowed = {"image/png", "image/jpeg", "image/bmp", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}. Use PNG/JPG/BMP.",
        )

    try:
        contents        = await file.read()
        tensor, vis_arr = preprocess_2d(contents)
        result          = tta_predict(model, tensor, vis_arr)
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/nifti", summary="Predict from NIfTI volume (.nii / .nii.gz)")
async def predict_nifti(file: UploadFile = File(...)):
    """
    Upload a NIfTI volume (.nii or .nii.gz).
    The middle axial slice is extracted and classified.
    Returns the classification result with probabilities and annotated image.
    - If TUMEUR: image with red bounding box around the tumor region (Grad-CAM).
    - If SAIN: plain image without annotation.
    The field `image_annotee` contains a base64-encoded PNG (data URI).
    """
    if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
        raise HTTPException(
            status_code=415,
            detail="Only .nii or .nii.gz files are accepted.",
        )

    try:
        contents        = await file.read()
        tensor, vis_arr = preprocess_nifti(contents)
        result          = tta_predict(model, tensor, vis_arr)
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("NIfTI prediction error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)