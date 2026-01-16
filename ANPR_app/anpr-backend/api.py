"""
License Plate Detection API with VGG19 Character Recognition

Pipeline: RT-DETRv2 → Plate Crop (OpenCV) → Character Segmentation (OpenCV) → VGG19 Recognition

Requirements:
pip install fastapi uvicorn python-multipart pillow torch transformers torchvision opencv-python numpy huggingface-hub
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import io
import base64
from typing import List, Dict, Tuple
import logging
import cv2
import numpy as np
import os
import tensorflow as tf
import keras  # The modern Keras 3 library

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
device = None

# Model configuration
MODEL_REPO = "ranm26/rtdetr-v2-r50-lpph-finetune-2"
VGG19_REPO = "ranm26/my-vgg19-classifier-v3"
CONFIDENCE_THRESHOLD = 0.25

# Character mapping (0-9, A-Z)
CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}


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
    """Load RT-DETRv2 and VGG19 models on startup"""
    global model, image_processor, char_model, device
    
    try:
        logger.info("Loading models...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load RT-DETRv2 for plate detection
        logger.info("Loading RT-DETRv2 model...")
        image_processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
        model = AutoModelForObjectDetection.from_pretrained(MODEL_REPO)
        model = model.to(device)
        model.eval()
        logger.info("✓ RT-DETRv2 model loaded successfully!")
        
        # Load VGG19 for character recognition
        logger.info(f"Loading VGG19 model from {VGG19_REPO}...")
        try:
            from huggingface_hub import hf_hub_download
            
            model_path = hf_hub_download(repo_id=VGG19_REPO, filename="my_vgg19_model-v3.keras")
            logger.info(f"Downloaded to: {model_path}")
            
            char_model = load_keras_vgg19_model(model_path)
            logger.info("✓ VGG19 character recognition model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading VGG19 model: {str(e)}")
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
        Binary image ready for contour detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def segment_characters(plate_image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Segment individual characters from license plate using OpenCV
    
    Args:
        plate_image: BGR image from OpenCV
        
    Returns:
        List of (character_image, bbox) tuples sorted left to right
    """
    # Preprocess
    binary = preprocess_plate_for_segmentation(plate_image)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours
    char_contours = []
    plate_height, plate_width = plate_image.shape[:2]
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size (characters should be significant portion of plate height)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Typical character constraints
        if (h > plate_height * 0.3 and  # Minimum height
            h < plate_height * 0.9 and  # Maximum height
            w > plate_width * 0.02 and   # Minimum width
            w < plate_width * 0.3 and    # Maximum width
            0.2 < aspect_ratio < 1.0):   # Aspect ratio
            
            # Extract character region with padding
            padding = 2
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(plate_width, x + w + padding)
            y2 = min(plate_height, y + h + padding)
            
            char_img = plate_image[y1:y2, x1:x2]
            char_contours.append((char_img, (x, y, w, h), x))  # Store x for sorting
    
    # Sort characters left to right
    char_contours.sort(key=lambda c: c[2])
    
    # Return character images and bboxes
    return [(char[0], char[1]) for char in char_contours]


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
        
        # Resize to model input size (128x128 for VGG19)
        char_image_resized = cv2.resize(char_image_rgb, (128, 128))
        
        # Normalize to [0, 1]
        char_image_normalized = char_image_resized.astype('float32') / 255.0
        
        # Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
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


def process_image(image: Image.Image, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> Dict:
    """
    Process image: RT-DETRv2 → Plate Crop → Character Segmentation → VGG19 Recognition
    
    Args:
        image: PIL Image object
        confidence_threshold: Minimum confidence for plate detection
        
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
            
            # Stage 2: Extract plate region (OpenCV)
            plate_pil = extract_plate_image(image, box_coords)
            plate_cv2 = cv2.cvtColor(np.array(plate_pil), cv2.COLOR_RGB2BGR)
            
            # Stage 3: Segment characters (OpenCV)
            character_segments = segment_characters(plate_cv2)
            
            # Stage 4: Recognize each character (VGG19)
            recognized_text = ""
            character_details = []
            avg_char_confidence = 0.0
            
            if char_model is not None and len(character_segments) > 0:
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
                
                logger.info(f"Plate text recognized: {recognized_text} (OCR confidence: {avg_char_confidence:.4f})")
            else:
                logger.warning("Character model not available or no character segments found")
            
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
                "num_characters": len(character_segments),
                "character_details": character_details,
                "vgg19_enabled": char_model is not None,
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
        "message": "License Plate Detection API with VGG19 OCR",
        "model": MODEL_REPO,
        "device": device,
        "vgg19_enabled": char_model is not None,
        "pipeline": "RT-DETRv2 → OpenCV Crop → OpenCV Segmentation → VGG19 Recognition"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "detection_model_loaded": model is not None,
        "character_model_loaded": char_model is not None,
        "device": device,
        "model_repo": MODEL_REPO
    }


@app.post("/detect")
async def detect_license_plate(
    file: UploadFile = File(...),
    confidence: float = CONFIDENCE_THRESHOLD
):
    """Detect license plates and recognize characters using VGG19"""
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
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: float = CONFIDENCE_THRESHOLD
):
    """Detect license plates in multiple images"""
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

