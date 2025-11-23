# Save this as 'convert_to_onnx.py'
import torch
import torch.nn as nn
from torchvision import transforms
import os

# --- 1. Define the Model Architecture (Copy your ResNet9 class here) ---
# ... (Paste your ResNet9 class and helper functions here) ...

# --- 2. Configuration and Model Loading ---
MODEL_PATH = './plant-disease-model-complete.pth'
ONNX_PATH = './plant-disease-model.onnx'
NUM_CLASSES = 38 
IN_CHANNELS = 3 

# Load the entire model object
if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
else:
    print(f"Error: PyTorch model file not found at {MODEL_PATH}")
    exit()

# --- 3. Export to ONNX ---
def export_onnx(model, onnx_path):
    # Create a dummy input tensor: (Batch Size=1, Channels=3, Height=256, Width=256)
    dummy_input = torch.randn(1, IN_CHANNELS, 256, 256, requires_grad=True)
    
    # Export the model
    torch.onnx.export(
        model,               # The model to be exported
        dummy_input,         # Model input (or a tuple for multiple inputs)
        onnx_path,           # Output file name
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=17,    # ONNX version to target (17 is a safe recent version)
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # The first dimension (batch size) is often dynamic for inference
        dynamic_axes={'input': {0: 'batch_size'}, 
                      'output': {0: 'batch_size'}}
    )
    print(f"Model successfully exported to {onnx_path}")

if __name__ == '__main__':
    export_onnx(model, ONNX_PATH)