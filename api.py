"""
License Plate Detection API with VGG19 or TROCR Character Recognition

Pipeline (VGG19): RT-DETRv2 → Plate Crop (OpenCV) → Character Segmentation (OpenCV) → VGG19 Recognition
Pipeline (TROCR): RT-DETRv2 → Plate Crop → TROCR Recognition (direct from plate image)

Requirements:
pip install fastapi uvicorn python-multipart pillow torch transformers torchvision opencv-python numpy huggingface-hub
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel, DetrImageProcessor, RTDetrV2ForObjectDetection
import io
import base64
from typing import List, Dict, Tuple, Optional
import logging
import cv2
import numpy as np
import os
try:
    import tensorflow as tf
    import keras  # The modern Keras 3 library
except ImportError:
    tf = None
    keras = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if tf is not None:
    tf.get_logger().setLevel('ERROR')

app = FastAPI(title="License Plate Detection API with VGG19 OCR")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
image_processor = None
char_model = None
trocr_processor = None
trocr_model = None
device = None

# Model configuration
MODEL_REPO = "ranm26/rtdetr-v2-r50-lpph-finetune-2"
VGG19_REPO = "ranm26/final-vgg19-v3"
TROCR_REPO = "microsoft/trocr-base-printed"
CONFIDENCE_THRESHOLD = 0.25

# Character mapping (0-9, A-Z)
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}

def build_vgg19_model():
    """Builds the exact model architecture used during training."""
    base_model = keras.applications.VGG19(
        weights=None, 
        include_top=False, 
        input_shape=(96, 96, 3)
    )
    
    model = keras.Sequential([
        keras.layers.Input(shape=(96, 96, 3)),
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(36, activation='softmax')
    ])
    return model

def load_keras_vgg19_model(model_path: str):
    """
    Load VGG19 model using the modern keras library.
    Uses compile=False to load successfully.
    Exits if model cannot be loaded.
    """
    logger.info(f"Loading model with keras.models.load_model(compile=False)...")
    try:
        # Load without compiling - this avoids the Flatten layer error
        model = keras.models.load_model(model_path, compile=False)
        logger.info("✓ Model loaded successfully!")
        return model
    except Exception as e:
        logger.error(f"✗ Load failed: {str(e)[:100]}")
        logger.error("FATAL: Could not load VGG19 model. Exiting.")
        exit(1)

@app.on_event("startup")
async def load_models():
    """Load RT-DETRv2 and both OCR models (VGG19 and TROCR) on startup"""
    global model, image_processor, char_model, trocr_processor, trocr_model, device
    
    try:
        logger.info("Loading models...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load RT-DETRv2 for plate detection
        logger.info("Loading RT-DETRv2 model...")
        try:
            # Try to load with AutoImageProcessor first
            image_processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
        except Exception as e:
            logger.warning(f"AutoImageProcessor failed: {str(e)}")
            logger.info("Trying with DetrImageProcessor...")
            # Fallback to DetrImageProcessor which is compatible with RT-DETRv2
            try:
                image_processor = DetrImageProcessor.from_pretrained(MODEL_REPO)
                logger.info("✓ DetrImageProcessor loaded successfully!")
            except Exception as e2:
                logger.error(f"DetrImageProcessor also failed: {str(e2)}")
                raise
        
        model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_REPO)
        model = model.to(device)
        model.eval()
        logger.info("✓ RT-DETRv2 model loaded successfully!")
        
        # Load VGG19 model
        logger.info(f"Loading VGG19 model from {VGG19_REPO}...")
        try:
            from huggingface_hub import hf_hub_download
            
            model_path = hf_hub_download(repo_id=VGG19_REPO, filename="vgg19_fixed_weights.weights.h5")
            logger.info(f"Downloaded to: {model_path}")
            
            char_model = build_vgg19_model()
            char_model.load_weights(model_path)
            logger.info("✓ VGG19 character recognition model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading VGG19 model: {str(e)}")
            raise
        
        # Load TROCR model
        logger.info(f"Loading TROCR model from {TROCR_REPO}...")
        try:
            trocr_processor = TrOCRProcessor.from_pretrained(TROCR_REPO)
            trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_REPO)
            trocr_model.to(device)
            trocr_model.eval()
            logger.info("✓ TROCR character recognition model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading TROCR model: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def preprocess_plate_for_segmentation(plate_image: np.ndarray) -> np.ndarray:
    """
    Preprocess license plate image for character segmentation
    
    Args:
        plate_image: BGR image from OpenCV
        
    Returns:
        Binary image ready for contour detection and ratio for spatial normalization when cropping each character
    """
    # Grayscale & Resize
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    target_h = 200
    ratio = target_h / gray.shape[0]
    gray = cv2.resize(gray, (int(gray.shape[1] * ratio), target_h), interpolation=cv2.INTER_CUBIC)

    # Normalize the contrast of the image
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # CLAHE utilization to enhance the image details
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Blur to merge broken character segments
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Convert into binary
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove the plate border
    h, w = binary.shape
    margin = int(h * 0.12)
    binary[0:margin, :] = 0
    binary[h-margin:h, :] = 0
    
    return binary, ratio


def segment_characters(plate_image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment individual characters from license plate using OpenCV
    
    Args:
        plate_image: BGR image from OpenCV
        
    Returns:
        List of (character_image, bbox) tuples sorted left to right
    """
    original_h, original_w = plate_image.shape[:2]
    binary, ratio = preprocess_plate_for_segmentation(plate_image)
    
    # Find all potential characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = binary.shape[:2]
    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Filter to catch blurry characters
        if h > (img_h * 0.2) and w < (img_w * 0.4) and aspect_ratio < 1.2:
            candidates.append((x, y, w, h))

    if not candidates:
        return []

    # Find the median from each character
    heights = [c[3] for c in candidates]
    median_h = np.median(heights)
    
    # 3. Filter by Height and Vertical alignment
    y_centers = [c[1] + (c[3] / 2) for c in candidates]
    median_y_center = np.median(y_centers)

    filtered = []
    for c in candidates:
        x, y, w, h = c
        center_y = y + (h / 2)
        
        if abs(h - median_h) < (median_h * 0.45) and abs(center_y - median_y_center) < (img_h * 0.15):
            filtered.append(c)

    # Sort from left to right
    filtered.sort(key=lambda x: x[0])

    # Prevent characters to be detected multiple times
    final_chars = []
    for i in range(len(filtered)):
        if i == 0:
            final_chars.append(filtered[i])
        else:
            prev_x = final_chars[-1][0]
            curr_x = filtered[i][0]
            if curr_x - prev_x > 5:
                final_chars.append(filtered[i])

    results = []
    for x, y, w, h in final_chars:
        # Return back to original size
        nx, ny, nw, nh = int(x/ratio), int(y/ratio), int(w/ratio), int(h/ratio)
        
        # Keep identified characters centered by adding padding
        p = 3
        x1, y1 = max(0, nx-p), max(0, ny-p)
        x2, y2 = min(original_w, nx+nw+p), min(original_h, ny+nh+p)
        
        char_crop = plate_image[y1:y2, x1:x2]
        if char_crop.size > 0:
            results.append((char_crop, (nx, ny, nw, nh)))
            
    return results


def recognize_character(char_image: np.ndarray, model, device: str) -> Tuple[str, float]:
    """
    Recognize a single character using VGG19 Keras model
    
    Args:
        char_image: BGR image of character from OpenCV
        model: VGG19 character recognition Keras model
        device: Device parameter (for compatibility, not used with Keras)
        
    Returns:
        (predicted_character, confidence)
    """
    try:
        # Validate input
        if char_image is None or char_image.size == 0:
            logger.warning("Empty character image received")
            return '?', 0.0
        
        # Convert BGR to RGB
        char_image_rgb = cv2.cvtColor(char_image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (96x96 for VGG19)
        char_image_resized = cv2.resize(char_image_rgb, (96, 96))
        
        # Normalize to [0, 1]
        char_image_normalized = char_image_resized.astype('float32') / 255.0
        
        # Add batch dimension: (96, 96, 3) -> (1, 96, 96, 3)
        input_data = np.expand_dims(char_image_normalized, axis=0)
        
        # Inference using Keras model
        predictions = model.predict(input_data, verbose=0)
        
        # Get the predicted class and confidence
        predicted_idx = int(np.argmax(predictions[0]))
        confidence_val = float(np.max(predictions[0]))
        
        # Map index to character (0-9, A-Z)
        predicted_char = IDX_TO_CHAR.get(predicted_idx, '?')
        
        logger.debug(f"Character recognition: {predicted_char} ({confidence_val:.4f})")
        
        return predicted_char, confidence_val
        
    except Exception as e:
        logger.error(f"Error recognizing character: {str(e)}")
        return '?', 0.0


def recognize_with_trocr(plate_image: Image.Image, processor, model, device: str) -> Tuple[str, float]:
    """
    Recognize text from license plate image using TROCR
    
    Args:
        plate_image: PIL Image of license plate
        processor: TrOCRProcessor instance
        model: TROCR model instance
        device: Device to run inference on
        
    Returns:
        (recognized_text, confidence)
    """
    try:
        # Validate input
        if plate_image is None:
            logger.warning("Empty plate image received")
            return '', 0.0
        
        # Preprocess the plate image (similar to trocr_alpr.py)
        # Convert to numpy for preprocessing
        img_array = np.array(plate_image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply adaptive thresholding to enhance text
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(denoised).convert('RGB')
        
        # Process image with TROCR processor
        pixel_values = processor(
            images=processed_image, 
            return_tensors="pt"
        ).pixel_values.to(device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        # Decode the generated text
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        # Post-process the text (remove spaces, convert to uppercase, common OCR fixes)
        cleaned_text = generated_text.replace(" ", "").upper()
        cleaned_text = cleaned_text.replace("O", "0")  # Common confusion
        cleaned_text = cleaned_text.replace("I", "1")  # Common confusion
        
        # Estimate confidence based on text length and character validity
        # Longer text with valid characters gets higher confidence
        if cleaned_text:
            valid_chars = sum(1 for c in cleaned_text if c in CHARACTERS)
            confidence = valid_chars / len(cleaned_text) if cleaned_text else 0.0
        else:
            confidence = 0.0
        
        logger.info(f"TROCR recognition: '{cleaned_text}' (confidence: {confidence:.4f})")
        
        return cleaned_text, confidence
        
    except Exception as e:
        logger.error(f"Error recognizing with TROCR: {str(e)}")
        return '', 0.0


def extract_plate_image(image: Image.Image, bbox: List[float], padding: int = 5) -> Image.Image:
    """Extract the license plate region from the image"""
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(image.width, int(x2) + padding)
    y2 = min(image.height, int(y2) + padding)
    return image.crop((x1, y1, x2, y2))


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def cv2_to_base64(cv2_image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', cv2_image)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_str}"


def process_image(image: Image.Image, confidence_threshold: float = CONFIDENCE_THRESHOLD, ocr_method: str = "VGG19") -> Dict:
    """
    Process image using specified OCR method:
    - VGG19: RT-DETRv2 → Plate Crop → Character Segmentation → VGG19 Recognition
    - TROCR: RT-DETRv2 → Plate Crop → TROCR Recognition (direct from plate image)
    
    Args:
        image: PIL Image object
        confidence_threshold: Minimum confidence for plate detection
        ocr_method: OCR method to use ("VGG19" or "TROCR")
        
    Returns:
        Dictionary with detection and recognition results
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        width, height = image.size
        
        # Stage 1: RT-DETRv2 Detection
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([[height, width]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=confidence_threshold, target_sizes=target_sizes
        )[0]
        
        detections = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_coords = [coord.item() for coord in box]
            
            # Stage 2: Extract plate region
            plate_pil = extract_plate_image(image, box_coords)
            
            # Stage 3 & 4: OCR based on specified method
            recognized_text = ""
            character_details = []
            avg_char_confidence = 0.0
            num_characters = 0
            
            if ocr_method == 'TROCR' and trocr_processor is not None and trocr_model is not None:
                # TROCR: Direct recognition from plate image (no character segmentation)
                recognized_text, avg_char_confidence = recognize_with_trocr(
                    plate_pil, trocr_processor, trocr_model, device
                )
                num_characters = len(recognized_text) if recognized_text else 0
                logger.info(f"Plate text recognized (TROCR): {recognized_text} (OCR confidence: {avg_char_confidence:.4f})")
                
            elif ocr_method == 'VGG19' and char_model is not None:
                # VGG19: Segment characters and recognize individually
                plate_cv2 = cv2.cvtColor(np.array(plate_pil), cv2.COLOR_RGB2BGR)
                character_segments = segment_characters(plate_cv2)
                num_characters = len(character_segments)
                
                if len(character_segments) > 0:
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
                    
                    avg_char_confidence = sum(char_confidences) / len(char_confidences) if char_confidences else 0.0
                    logger.info(f"Plate text recognized (VGG19): {recognized_text} (OCR confidence: {avg_char_confidence:.4f})")
                else:
                    logger.warning("No character segments found")
            else:
                logger.warning(f"OCR model not available for method: {ocr_method}")
            
            detection = {
                "label": model.config.id2label[label.item()],
                "confidence": round(score.item(), 4),
                "bbox": {
                    "x1": round(box_coords[0], 2),
                    "y1": round(box_coords[1], 2),
                    "x2": round(box_coords[2], 2),
                    "y2": round(box_coords[3], 2),
                    "width": round(box_coords[2] - box_coords[0], 2),
                    "height": round(box_coords[3] - box_coords[1], 2)
                },
                "plate_image": image_to_base64(plate_pil),
                "plate_text": recognized_text if recognized_text else "UNREADABLE",
                "ocr_confidence": round(avg_char_confidence, 4),
                "num_characters": num_characters,
                "character_details": character_details,
                "ocr_method": ocr_method,
                "is_readable": len(recognized_text) > 0 and avg_char_confidence > 0.5
            }
            detections.append(detection)
            logger.info(f"Detection completed: {detection['plate_text']}")
        
        return {
            "success": True,
            "num_detections": len(detections),
            "image_dimensions": {"width": width, "height": height},
            "detections": detections
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "License Plate Detection API with VGG19 and TROCR OCR",
        "model": MODEL_REPO,
        "device": device,
        "vgg19_loaded": char_model is not None,
        "trocr_loaded": trocr_model is not None,
        "pipeline_vgg19": "RT-DETRv2 → Plate Crop → Character Segmentation → VGG19 Recognition",
        "pipeline_trocr": "RT-DETRv2 → Plate Crop → TROCR Recognition (direct)"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "detection_model_loaded": model is not None,
        "vgg19_loaded": char_model is not None,
        "trocr_loaded": trocr_model is not None,
        "device": device,
        "model_repo": MODEL_REPO
    }


@app.post("/detect")
async def detect_license_plate(
    file: UploadFile = File(...),
    confidence: float = CONFIDENCE_THRESHOLD,
    ocr_method: str = "VGG19"
):
    """Detect license plates and recognize characters using specified OCR method (VGG19 or TROCR)"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
        
        if ocr_method not in ["VGG19", "TROCR"]:
            raise HTTPException(status_code=400, detail="OCR method must be 'VGG19' or 'TROCR'")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        results = process_image(image, confidence_threshold=confidence, ocr_method=ocr_method)
        
        return JSONResponse(content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: float = CONFIDENCE_THRESHOLD,
    ocr_method: str = "VGG19"
):
    """Detect license plates in multiple images using specified OCR method (VGG19 or TROCR)"""
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
        
        if ocr_method not in ["VGG19", "TROCR"]:
            raise HTTPException(status_code=400, detail="OCR method must be 'VGG19' or 'TROCR'")
        
        results = []
        
        for file in files:
            try:
                if not file.content_type.startswith("image/"):
                    results.append({"filename": file.filename, "success": False, "error": "Invalid file type"})
                    continue
                
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                detection_result = process_image(image, confidence_threshold=confidence, ocr_method=ocr_method)
                detection_result["filename"] = file.filename
                results.append(detection_result)
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({"filename": file.filename, "success": False, "error": str(e)})
        
        return JSONResponse(content={"success": True, "total_images": len(files), "results": results})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch detect endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
