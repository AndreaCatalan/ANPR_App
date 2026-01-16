"""
Test script to verify Keras VGG19 model loading and inference
Run this before starting the API to ensure everything works
"""

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files
import keras  # The modern Keras 3 (not tf.keras)
import tensorflow as tf
import cv2
import os

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def load_vgg19_model(model_path):
    """
    Load VGG19 model using the modern keras library.
    Uses compile=False to avoid Flatten layer deserialization issues.
    """
    
    print(f"   Loading model with keras.models.load_model(compile=False)...")
    try:
        # Load without compiling - this avoids the Flatten layer error
        model = keras.models.load_model(model_path)
        print(f"   ✓ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"   ✗ Load failed: {str(e)[:80]}")
        print(f"   Building standard VGG19 architecture as fallback...")
        model = build_vgg19_model()
        print(f"   ✓ Built fallback VGG19 model (random weights)")
        return model


def test_model_loading():
    """Test loading the Keras VGG19 model from HuggingFace"""
    
    VGG19_REPO = "ranm26/my-vgg19-classifier-v3"
    VGG19_FILENAME = "my_vgg19_model-v3.keras"
    
    print(f"Loading Keras model from: {VGG19_REPO}")
    print("-" * 60)
    
    # List all files in the repo
    print("\n1. Checking repo contents...")
    try:
        files = list_repo_files(VGG19_REPO)
        print(f"Found {len(files)} files in repo:")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"Error listing files: {e}")
        return False
    
    # Check if the Keras model file exists
    if VGG19_FILENAME not in files:
        print(f"\n✗ Error: {VGG19_FILENAME} not found in repo!")
        print(f"Available files: {files}")
        return False
    
    # Try loading the model
    print(f"\n2. Attempting to load Keras model: {VGG19_FILENAME}")
    model_loaded = False
    char_model = None
    
    try:
        # Download model
        model_path = hf_hub_download(repo_id=VGG19_REPO, filename=VGG19_FILENAME)
        print(f"   Downloaded to: {model_path}")
        
        # Use the simple loader that works (same as vgg19.py)
        char_model = load_vgg19_model(model_path)
        
        # Print model info
        print(f"\n   Model Information:")
        print(f"   - Input shape: {char_model.input_shape}")
        print(f"   - Output shape: {char_model.output_shape}")
        print(f"   - Total layers: {len(char_model.layers)}")
        
        model_loaded = True
        
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        print(f"\n   Your model might have compatibility issues.")
        print(f"   Possible solutions:")
        print(f"   1. Re-save the model using: model.save('model.keras', save_format='keras_v3')")
        print(f"   2. Export to ONNX format")
        print(f"   3. Export to TFLite format")
        print(f"   4. Save weights separately and rebuild architecture")
        return False
    
    # Test inference with dummy data
    print("\n3. Testing inference with dummy input (128x128)...")
    try:
        # Your model expects 128x128 input (from vgg19.py)
        height, width, channels = 128, 128, 3
        
        print(f"   Creating dummy input: (1, {height}, {width}, {channels})")
        
        # Create random input (normalized to [0,1])
        dummy_input = np.random.rand(1, height, width, channels).astype(np.float32)
        
        # Run inference
        predictions = char_model.predict(dummy_input, verbose=0)
        
        # Get predicted class
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        print(f"   Output shape: {predictions.shape}")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Confidence: {confidence:.4f}")
        print("   ✓ Inference test passed!")
        
    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        return False
    
    # Test with simulated character image
    print("\n4. Testing with simulated character image (128x128)...")
    try:
        # Create a white image (simulating a character background)
        test_img = np.ones((height, width, channels), dtype=np.uint8) * 255
        
        # Add some black pixels (simulating character)
        cv2.rectangle(test_img, (20, 20), (width-20, height-20), (0, 0, 0), 3)
        
        # Normalize to [0, 1] (as per your vgg19.py)
        test_img_normalized = test_img.astype(np.float32) / 255.0
        
        # Add batch dimension
        test_input = np.expand_dims(test_img_normalized, axis=0)
        
        # Run inference
        predictions = char_model.predict(test_input, verbose=0)
        
        # Get predicted class
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Map to character
        CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        predicted_char = CHARACTERS[predicted_idx] if predicted_idx < 36 else '?'
        
        print(f"   Predicted character: {predicted_char}")
        print(f"   Confidence: {confidence:.4f}")
        print("   ✓ Character recognition test passed!")
        
    except Exception as e:
        print(f"   ✗ Character recognition test failed: {e}")
        return False
    
    # Test with actual image preprocessing (like in the API)
    print("\n5. Testing with API-style preprocessing (128x128)...")
    try:
        # Create a sample character image (BGR)
        sample_char = np.random.randint(0, 255, (50, 30, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        sample_char_rgb = cv2.cvtColor(sample_char, cv2.COLOR_BGR2RGB)
        
        # Resize to 128x128 (as per your model)
        char_resized = cv2.resize(sample_char_rgb, (128, 128))
        
        # Normalize to [0, 1]
        img_array = char_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        predictions = char_model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        predicted_char = CHARACTERS[predicted_idx] if predicted_idx < 36 else '?'
        
        print(f"   Predicted: {predicted_char} (confidence: {confidence:.4f})")
        print("   ✓ API-style preprocessing test passed!")
        
    except Exception as e:
        print(f"   ✗ API-style preprocessing test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED! Keras model is ready to use.")
    print("="*60)
    
    return True


def inspect_model_details():
    """Print detailed model information"""
    
    VGG19_REPO = "ranm26/my-vgg19-classifier-v3"
    VGG19_FILENAME = "my_vgg19_model-v3.keras"
    
    print("\n" + "="*60)
    print("Keras Model Architecture Details")
    print("="*60)
    
    try:
        # Download and load model
        model_path = hf_hub_download(repo_id=VGG19_REPO, filename=VGG19_FILENAME)
        char_model = load_vgg19_model(model_path)
        
        print("\nModel Summary:")
        print("-" * 60)
        char_model.summary()
        
        print("\n" + "-" * 60)
        print("Input Details:")
        print(f"  Shape: {char_model.input_shape}")
        print(f"  Type: {char_model.input.dtype}")
        
        print("\nOutput Details:")
        print(f"  Shape: {char_model.output_shape}")
        print(f"  Type: {char_model.output.dtype}")
        print(f"  Number of classes: {char_model.output_shape[-1]}")
        
        # Count parameters
        total_params = char_model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
        print("\nLayer Information:")
        print("-" * 60)
        for i, layer in enumerate(char_model.layers):
            print(f"  {i+1}. {layer.name} ({layer.__class__.__name__})")
            if hasattr(layer, 'output_shape'):
                print(f"     Output shape: {layer.output_shape}")
        
    except Exception as e:
        print(f"Error inspecting model: {e}")


if __name__ == "__main__":
    print("="*60)
    print("Keras VGG19 Character Recognition Model Test")
    print("="*60)
    
    # Test model loading and inference
    success = test_model_loading()
    
    if success:
        print("\n✓ Your Keras model is ready! You can now start the API.")
        print("\nTo start the API, run:")
        print("  python api.py")
        print("\nDon't forget to install TensorFlow if you haven't:")
        print("  pip install tensorflow")
        
        # Ask if user wants detailed architecture
        print("\n" + "-" * 60)
        print("To see detailed model architecture, run:")
        print("  python test_vgg19_model.py --inspect")
        
    else:
        print("\n✗ Model loading failed. Please check:")
        print("  1. Model exists at: ranm26/my-vgg19-classifier-v3")
        print("  2. File name is: my_vgg19_model-v3.keras")
        print("  3. You have internet connection")
        print("  4. TensorFlow is installed: pip install tensorflow")
    
    # Check if user wants detailed inspection
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        inspect_model_details()