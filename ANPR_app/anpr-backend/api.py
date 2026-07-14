from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, DetrImageProcessor, RTDetrV2ForObjectDetection
import io
import base64
from typing import List, Dict, Tuple, Optional
import logging
import cv2
import numpy as np
import os
try:
    import tensorflow as tf
    import keras
except ImportError:
    tf = None
    keras = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if tf is not None:
    tf.get_logger().setLevel('ERROR')

app = FastAPI(title="License Plate Detection API with VGG19 OCR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

model = None
image_processor = None
char_model = None
device = None

MODEL_REPO = "ranm26/rtdetr-v2-setup8"
VGG19_REPO = "ranm26/my-vgg19-classifier-v3"
CONFIDENCE_THRESHOLD = 0.25

CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}

import keras


@app.on_event("startup")
async def load_models():
    global model, image_processor, char_model, device
    try:
        logger.info("Loading models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # RT-DETRv2
        logger.info("Loading RT-DETRv2 model...")
        try:
            image_processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
        except Exception:
            logger.warning("AutoImageProcessor failed, trying DetrImageProcessor...")
            image_processor = DetrImageProcessor.from_pretrained(MODEL_REPO)
        model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_REPO)
        model = model.to(device)
        model.eval()
        logger.info("✓ RT-DETRv2 loaded!")

        # VGG19
        logger.info(f"Loading VGG19 from {VGG19_REPO}...")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=VGG19_REPO, filename="vgg19_v4_final.keras")
        char_model = keras.models.load_model(model_path, compile=False)
        logger.info("✓ VGG19 loaded!")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


# DARK/LOW-LIGHT CONDITION HANDLING
def enhance_dark_plate(plate_image: np.ndarray) -> np.ndarray:
    """Enhance dark/night plates before segmentation."""
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness > 120:
        return plate_image

    logger.info(f"Dark plate (brightness={mean_brightness:.1f}), enhancing...")

    if mean_brightness < 40:
        logger.info("Extreme darkness detected - applying aggressive enhancement")
        gamma = 4.5
        table = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype(np.uint8)
        plate_image = cv2.LUT(plate_image, table)
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        logger.info(f"After initial boost: brightness={mean_brightness:.1f}")

    gamma = 4.0 if mean_brightness < 50 else (3.0 if mean_brightness < 80 else 2.0)
    table = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype(np.uint8)
    plate_image = cv2.LUT(plate_image, table)

    lab = cv2.cvtColor(plate_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    plate_image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    plate_image = cv2.normalize(plate_image, None, 0, 255, cv2.NORM_MINMAX)
    return plate_image


# ANGLED VIEW / PERSPECTIVE CORRECTION
def deskew_plate(plate_image: np.ndarray) -> np.ndarray:
    """Correct plate angle using Hough line detection."""
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        angles = [np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])*180/np.pi for l in lines]
        valid = [a for a in angles if abs(a) < 45]
        if valid:
            angle = np.median(valid)
            h, w = plate_image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            return cv2.warpAffine(plate_image, M, (w, h),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return plate_image


# MULTI-STRATEGY BINARIZATION
def get_best_binary(gray: np.ndarray) -> np.ndarray:
    """Try multiple binarization strategies, return the one with most valid contours."""
    h, w = gray.shape
    candidates = []

    # Strategy 1: Otsu on blurred
    b1 = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                       cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    candidates.append(b1)

    # Strategy 2: CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    b2 = cv2.threshold(cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0), 0, 255,
                       cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    candidates.append(b2)

    # Strategy 3: Adaptive Gaussian
    b3 = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    candidates.append(b3)

    # Strategy 4: Equalized + Otsu
    b4 = cv2.threshold(cv2.GaussianBlur(cv2.equalizeHist(gray), (5, 5), 0), 0, 255,
                       cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    candidates.append(b4)

    # Strategy 5 (NEW): Morphological opening after Otsu to separate touching/bleeding chars.
    # Targets blurry/low-contrast plates where ink bleeds into neighbors (e.g. C→P, B+G merge).
    # A small vertical-kernel open breaks horizontal ink bridges between close characters
    # without destroying the vertical stroke body of each character.
    raw_otsu = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
                              cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    sep_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  # vertical only — preserves char width
    b5 = cv2.morphologyEx(raw_otsu, cv2.MORPH_OPEN, sep_kernel)
    candidates.append(b5)

    margin = int(h * 0.12)
    for b in candidates:
        b[0:margin, :] = 0
        b[h-margin:h, :] = 0

    def score(binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            ar = cw/float(ch) if ch > 0 else 0
            if (h*0.20 < ch < h*0.90 and w*0.01 < cw < w*0.30 and 0.05 < ar < 1.5):
                count += 1
        return count

    best = max(candidates, key=score)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(best, cv2.MORPH_CLOSE, kernel)


# CHARACTER CONTOUR EXTRACTION
def extract_char_contours(binary: np.ndarray, plate_image: np.ndarray,
                           min_h=0.20, max_h=0.90, min_w=0.01, max_w=0.30,
                           min_ar=0.05, max_ar=1.5, h_diff=0.50, y_diff=0.20):
    """Extract and filter character contours from binary image."""
    img_h, img_w = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w/float(h) if h > 0 else 0
        if (img_h*min_h < h < img_h*max_h and
                img_w*min_w < w < img_w*max_w and
                min_ar < ar < max_ar):
            candidates.append((x, y, w, h))

    if not candidates:
        return []

    heights = [c[3] for c in candidates]
    med_h = np.median(heights)
    y_centers = [c[1] + c[3]/2 for c in candidates]
    med_y = np.median(y_centers)

    filtered = [(x, y, w, h) for x, y, w, h in candidates
                if abs(h - med_h) < med_h*h_diff
                and abs((y+h/2) - med_y) < img_h*y_diff]
    filtered.sort(key=lambda c: c[0])

    # Estimate typical character width for smarter adaptive thresholds.
    # Using median width instead of a fixed fraction of plate width makes
    # the gap and merge logic scale correctly regardless of how many
    # characters are on the plate or how zoomed-in the crop is.
    widths = [c[2] for c in filtered]
    med_w = np.median(widths) if widths else img_w * 0.08

    result = []
    for c in filtered:
        if not result:
            result.append(list(c))
        else:
            prev = result[-1]
            overlap = (prev[0] + prev[2]) - c[0]
            smaller_w = min(prev[2], c[2])

            # FIX 1: Tighten overlap merge threshold from 0.60 → 0.75.
            # The original 0.60 was too permissive: closely spaced characters
            # like "B" and "G" were being merged into a single wide contour,
            # causing one character to silently disappear.
            # At 0.75 we only merge genuine duplicate/split contours of the
            # same character (true overlap > 75%), not merely adjacent chars.
            if overlap > smaller_w * 0.75:
                nx = min(prev[0], c[0])
                nw = max(prev[0]+prev[2], c[0]+c[2]) - nx
                result[-1] = [nx, prev[1], nw, max(prev[3], c[3])]

            # FIX 2: Replace the plate-width-relative gap threshold with a
            # median-char-width-relative one.
            # Old: img_w * 0.40  ← nearly half the plate, swallowed the
            #      CBG → 1042 separator gap and merged the two groups.
            # New: med_w * 1.8   ← 1.8× the typical char width comfortably
            #      covers normal inter-character spacing (and even the
            #      separator hyphen space) without bridging across the
            #      larger group-to-group gap on PH-format plates.
            elif c[0] - (prev[0] + prev[2]) < med_w * 1.8:
                result.append(list(c))
            else:
                # Gap is larger than 1.8 char widths — still keep the
                # character rather than silently dropping it.
                result.append(list(c))

    chars = []
    ph, pw = plate_image.shape[:2]
    for x, y, w, h in result:
        pad = 3
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(pw, x+w+pad); y2 = min(ph, y+h+pad)
        img = plate_image[y1:y2, x1:x2]
        if img.size > 0:
            chars.append((img, (x, y, w, h)))
    return chars


# MAIN SEGMENTATION PIPELINE
def segment_characters(plate_image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Multi-strategy character segmentation with automatic fallbacks."""
    plate_image = enhance_dark_plate(plate_image)
    plate_image = deskew_plate(plate_image)

    h, w = plate_image.shape[:2]
    if w < 300:
        scale = 300/w
        plate_image = cv2.resize(plate_image, (int(w*scale), int(h*scale)),
                                 interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    binary = get_best_binary(gray)

    param_sets = [
        dict(min_h=0.20, max_h=0.90, min_w=0.01, max_w=0.30, min_ar=0.05, max_ar=1.5, h_diff=0.45, y_diff=0.15),
        dict(min_h=0.15, max_h=0.92, min_w=0.008, max_w=0.35, min_ar=0.04, max_ar=1.8, h_diff=0.55, y_diff=0.20),
        dict(min_h=0.10, max_h=0.95, min_w=0.005, max_w=0.40, min_ar=0.03, max_ar=2.0, h_diff=0.70, y_diff=0.30),
    ]

    for i, params in enumerate(param_sets):
        result = extract_char_contours(binary, plate_image, **params)
        if len(result) >= 4:
            logger.info(f"Contour segmentation (pass {i+1}): {len(result)} chars")
            return result
        logger.info(f"Pass {i+1}: {len(result)} chars, trying looser params...")

    logger.info("Contour segmentation failed — returning best attempt")
    result = extract_char_contours(binary, plate_image,
                                   min_h=0.08, max_h=0.95, min_w=0.004, max_w=0.45,
                                   min_ar=0.02, max_ar=2.5, h_diff=0.80, y_diff=0.40)
    return result


# CHARACTER RECOGNITION
def recognize_character(char_image: np.ndarray, model, device: str) -> Tuple[str, float]:
    try:
        if char_image is None or char_image.size == 0:
            return '?', 0.0

        char_image_rgb = cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB)

        # Color normalization - remove sepia/tints
        lab = cv2.cvtColor(char_image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        color_deviation = abs(a_mean - 128) + abs(b_mean - 128)
        if color_deviation > 20:
            logger.info(f"Color tint detected (deviation={color_deviation:.1f}), normalizing...")
            a = np.full_like(a, 128)
            b = np.full_like(b, 128)
            lab = cv2.merge((l, a, b))
            char_image_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Adaptive gamma correction
        gray = cv2.cvtColor(char_image_rgb, cv2.COLOR_RGB2GRAY)
        mean_bright = np.mean(gray)
        if mean_bright < 40:
            gamma = 2.5
        elif mean_bright < 80:
            gamma = 1.8
        elif mean_bright < 120:
            gamma = 1.4
        else:
            gamma = None

        if gamma:
            table = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype(np.uint8)
            char_image_rgb = cv2.LUT(char_image_rgb, table)

        # CLAHE + contrast normalization
        lab = cv2.cvtColor(char_image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        l = clahe.apply(l)
        a = cv2.normalize(a, None, 100, 155, cv2.NORM_MINMAX)
        b = cv2.normalize(b, None, 100, 155, cv2.NORM_MINMAX)
        lab = cv2.merge((l, a, b))
        char_image_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # VGG19 inference
        char_image_resized = cv2.resize(char_image_rgb, (128, 128))
        char_image_normalized = char_image_resized.astype('float32') / 255.0
        input_data = np.expand_dims(char_image_normalized, axis=0)
        predictions = model.predict(input_data, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence_val = float(np.max(predictions[0]))
        predicted_char = IDX_TO_CHAR.get(predicted_idx, '?')
        return predicted_char, confidence_val
    except Exception as e:
        logger.error(f"Error recognizing character: {str(e)}")
        return '?', 0.0


# UTILITIES
def extract_plate_image(image: Image.Image, bbox: List[float], padding: int = 5) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)-padding); y1 = max(0, int(y1)-padding)
    x2 = min(image.width, int(x2)+padding); y2 = min(image.height, int(y2)+padding)
    return image.crop((x1, y1, x2, y2))

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def cv2_to_base64(cv2_image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', cv2_image)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"


# MAIN PROCESSING PIPELINE
def process_image(image: Image.Image, confidence_threshold: float = CONFIDENCE_THRESHOLD,
                  ocr_method: str = "VGG19") -> Dict:
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size

        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes)[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_coords = [coord.item() for coord in box]
            plate_pil = extract_plate_image(image, box_coords)
            recognized_text = ""
            character_details = []
            avg_char_confidence = 0.0
            num_characters = 0

            if char_model is not None:
                plate_cv2 = cv2.cvtColor(np.array(plate_pil), cv2.COLOR_RGB2BGR)
                character_segments = segment_characters(plate_cv2)
                num_characters = len(character_segments)

                if len(character_segments) >= 4:
                    char_confidences = []
                    for char_img, char_bbox in character_segments:
                        char, conf = recognize_character(char_img, char_model, device)
                        recognized_text += char
                        char_confidences.append(conf)
                        character_details.append({
                            "character": char,
                            "confidence": round(conf, 4),
                            "bbox": list(char_bbox) if isinstance(char_bbox, tuple) else char_bbox,
                            "image": cv2_to_base64(char_img)
                        })
                    avg_char_confidence = sum(char_confidences)/len(char_confidences) if char_confidences else 0.0
                    logger.info(f"VGG19: {recognized_text} ({avg_char_confidence:.4f})")
                else:
                    logger.warning(f"Segmentation found only {len(character_segments)} chars — plate may be UNREADABLE")
                    if character_segments:
                        for char_img, char_bbox in character_segments:
                            char, conf = recognize_character(char_img, char_model, device)
                            recognized_text += char
                            character_details.append({
                                "character": char,
                                "confidence": round(conf, 4),
                                "bbox": list(char_bbox) if isinstance(char_bbox, tuple) else char_bbox,
                                "image": cv2_to_base64(char_img)
                            })
            else:
                logger.warning("VGG19 model not available")

            detection = {
                "label": model.config.id2label[label.item()],
                "confidence": round(score.item(), 4),
                "bbox": {
                    "x1": round(box_coords[0], 2), "y1": round(box_coords[1], 2),
                    "x2": round(box_coords[2], 2), "y2": round(box_coords[3], 2),
                    "width": round(box_coords[2]-box_coords[0], 2),
                    "height": round(box_coords[3]-box_coords[1], 2)
                },
                "plate_image": image_to_base64(plate_pil),
                "plate_text": recognized_text if recognized_text else "UNREADABLE",
                "ocr_confidence": round(avg_char_confidence, 4),
                "num_characters": num_characters,
                "character_details": character_details,
                "ocr_method": "VGG19",
                "is_readable": len(recognized_text) > 0 and avg_char_confidence > 0.5
            }
            detections.append(detection)

        return {
            "success": True,
            "num_detections": len(detections),
            "image_dimensions": {"width": width, "height": height},
            "detections": detections
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


# ENDPOINTS
@app.get("/")
async def root():
    return {"status": "ok", "message": "License Plate Detection API",
            "device": device, "vgg19_loaded": char_model is not None}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "detection_model_loaded": model is not None,
            "vgg19_loaded": char_model is not None, "device": device}


@app.post("/detect")
async def detect_license_plate(file: UploadFile = File(...),
                                confidence: float = CONFIDENCE_THRESHOLD):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        results = process_image(image, confidence_threshold=confidence)
        return JSONResponse(content=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...),
                       confidence: float = CONFIDENCE_THRESHOLD):
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
        results = []
        for file in files:
            try:
                if not file.content_type.startswith("image/"):
                    results.append({"filename": file.filename, "success": False, "error": "Invalid file type"})
                    continue
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                detection_result = process_image(image, confidence_threshold=confidence)
                detection_result["filename"] = file.filename
                results.append(detection_result)
            except Exception as e:
                results.append({"filename": file.filename, "success": False, "error": str(e)})
        return JSONResponse(content={"success": True, "total_images": len(files), "results": results})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)