# ANPR Backend - License Plate Detection API

FastAPI-based backend service for the Automatic Number Plate Recognition (ANPR) system. This service provides RESTful API endpoints for detecting license plates in images and recognizing characters using deep learning models.

## Features

- **License Plate Detection**: Uses RT-DETRv2 model for accurate plate detection
- **Character Recognition**: VGG19-based OCR for character recognition
- **Character Segmentation**: OpenCV-based character segmentation from detected plates
- **Batch Processing**: Support for processing multiple images
- **Real-time Processing**: Optimized for fast inference
- **CORS Support**: Configured for React frontend integration

## Architecture

### Pipeline
```
Image → RT-DETRv2 Detection → Plate Crop (OpenCV) → Character Segmentation (OpenCV) → VGG19 Recognition → Results
```

### Models
- **Detection Model**: `ranm26/rtdetr-v2-r50-lpph-finetune-2` (RT-DETRv2)
- **Character Recognition Model**: `ranm26/my-vgg19-classifier-v3` (VGG19)

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster inference

## Installation

### 1. Navigate to the backend directory
```bash
cd anpr-backend
```

### 2. Create and activate virtual environment (recommended)
```bash
# Create virtual environment
python -m venv be

# Activate on Linux/Mac
source be/bin/activate

# Activate on Windows (Command Prompt)
be\Scripts\activate.bat

# Activate on Windows (PowerShell)
be\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Note**: The requirements include:
- `fastapi==0.104.1` - Web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `torch==2.1.1` - Deep learning framework
- `transformers==4.36.0` - Hugging Face models
- `opencv-python==4.8.1.78` - Image processing
- `pillow==10.1.0` - Image handling
- `tensorflow==2.15.0` - Keras/VGG19 support
- `keras==3.0.5` - Modern Keras library

### 4. Verify GPU Availability (Optional)
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is not available, the service will automatically use CPU (slower but functional).

## Running the Backend

### Development Mode

Start the server with auto-reload:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

Start the server without reload:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Using Python Directly

Alternatively, you can run:
```bash
python api.py
```

The server will be available at `http://localhost:8000`.

## API Endpoints

### GET /
**Description**: API information and status

**Response**:
```json
{
  "status": "ok",
  "message": "License Plate Detection API with VGG19 OCR",
  "model": "ranm26/rtdetr-v2-r50-lpph-finetune-2",
  "device": "cuda" | "cpu",
  "vgg19_enabled": true,
  "pipeline": "RT-DETRv2 → OpenCV Crop → OpenCV Segmentation → VGG19 Recognition"
}
```

### GET /health
**Description**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "detection_model_loaded": true,
  "character_model_loaded": true,
  "device": "cuda" | "cpu",
  "model_repo": "ranm26/rtdetr-v2-r50-lpph-finetune-2"
}
```

### POST /detect
**Description**: Detect license plates and recognize characters

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (image/*)
- Query Parameter: `confidence` (optional, default: 0.25) - Detection confidence threshold (0.0 to 1.0)

**Response**:
```json
{
  "success": true,
  "num_detections": 1,
  "image_dimensions": {
    "width": 1920,
    "height": 1080
  },
  "detections": [
    {
      "label": "license plate",
      "confidence": 0.95,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 300.7,
        "y2": 250.9,
        "width": 200.2,
        "height": 50.6
      },
      "plate_image": "data:image/png;base64,...",
      "plate_text": "ABC123",
      "ocr_confidence": 0.89,
      "num_characters": 6,
      "character_details": [
        {
          "character": "A",
          "confidence": 0.92,
          "bbox": [105, 205, 120, 240],
          "image": "data:image/png;base64,..."
        }
      ],
      "vgg19_enabled": true,
      "is_readable": true
    }
  ]
}
```

### POST /detect/batch
**Description**: Process multiple images in batch

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: Multiple image files (up to 10 images)
- Query Parameter: `confidence` (optional, default: 0.25) - Detection confidence threshold

**Response**:
```json
{
  "success": true,
  "total_images": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "num_detections": 1,
      "detections": [...]
    }
  ]
}
```

## Testing the API

### Using cURL

**Single image detection**:
```bash
curl -X POST "http://localhost:8000/detect?confidence=0.3" \
  -F "file=@/path/to/image.jpg"
```

**Health check**:
```bash
curl http://localhost:8000/health
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Detect license plate
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/detect", files=files)
    print(response.json())
```

### Using the Frontend

1. Start the frontend (see anpr-app README)
2. Upload an image through the web interface
3. Results will be displayed automatically

## Project Structure

```
anpr-backend/
├── api.py                    # Main FastAPI application
├── requirements.txt          # Python dependencies
├── test_vgg19_model.py       # Model testing script
├── legacy_vgg19.h5           # Legacy model file (if needed)
├── be/                       # Virtual environment
│   ├── bin/
│   ├── lib/
│   └── ...
└── README.md                 # This file
```

## Model Loading

On startup, the application automatically:
1. Downloads RT-DETRv2 model from Hugging Face
2. Downloads VGG19 model from Hugging Face
3. Loads models into memory
4. Detects available device (CUDA/CPU)

**First run may take several minutes** due to model downloads. Subsequent runs will use cached models.

## Performance Considerations

### GPU vs CPU
- **GPU (CUDA)**: Recommended for production, ~10x faster
- **CPU**: Functional but slower, suitable for development/testing

### Memory Requirements
- RT-DETRv2: ~2GB VRAM
- VGG19: ~500MB VRAM
- Total: ~2.5GB VRAM recommended

### Batch Processing
- Maximum 10 images per batch request
- Larger batches may cause timeouts or memory issues

## Troubleshooting

### Model Download Failures
If models fail to download:
1. Check internet connection
2. Verify Hugging Face access
3. Check disk space (models are ~500MB total)
4. Set `HF_TOKEN` environment variable if needed

### CUDA Out of Memory
If you get CUDA OOM errors:
1. Reduce batch size
2. Use CPU mode by setting `device = "cpu"` in `api.py`
3. Close other GPU applications

### Slow Inference on CPU
- Expected: 2-5 seconds per image
- If slower: Check system resources, close unnecessary applications

### Port Already in Use
```bash
# Kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn api:app --host 0.0.0.0 --port 8001
```

### CORS Issues
The backend is configured with permissive CORS for development. For production, update the `allow_origins` list in `api.py`:
```python
allow_origins=["http://localhost:5173", "https://your-production-domain.com"]
```

### Keras Model Loading Error
If you encounter errors loading the VGG19 model:
1. Ensure TensorFlow and Keras are properly installed
2. Check that the model file exists in the Hugging Face repository
3. Verify the model format is compatible (`.keras` format)

## Development

### Adding New Endpoints
1. Add new route functions in `api.py`
2. Use FastAPI decorators: `@app.get()`, `@app.post()`
3. Add type hints for better documentation
4. Update this README with new endpoint documentation

### Modifying Models
1. Update model repository names in `api.py`
2. Adjust preprocessing/postprocessing as needed
3. Test with `test_vgg19_model.py`

### Logging
The application uses Python's logging module:
- INFO: Model loading, processing status
- WARNING: Non-critical issues
- ERROR: Critical failures

View logs in the terminal where the server is running.

## Security Notes

- The current CORS configuration allows all origins (`*`). **Change this for production**.
- File uploads are not validated beyond content type checking.
- No authentication/authorization is implemented.
- Consider adding rate limiting for production deployments.

## License

This project is part of the ANPR system thesis project.

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RT-DETRv2 Paper](https://arxiv.org/abs/2304.08069)
- [VGG19 Architecture](https://arxiv.org/abs/1409.1556)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
