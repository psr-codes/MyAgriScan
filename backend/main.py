


# model_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# --- 1. Define Model Architecture (MUST BE EXACT COPY FROM YOUR NOTEBOOK) ---

# Helper function (usually needed for ResNet implementation)
def conv_block(in_channels, out_channels, pool=False):
    # COPY THIS FUNCTION EXACTLY
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    # COPY THIS CLASS EXACTLY
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64) 
        self.conv2 = conv_block(64, 128, pool=True) 
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4), 
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# --- 2. Configuration (The 38 PlantVillage Classes) ---

# This list must match the order of classes used when training your model!
CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 3. Preprocessing and Inference Logic ---

def transform_image(image_bytes):
    """
    Transforms the raw image bytes into the PyTorch tensor format required by the model.
    """
    # Define transformations matching the training data (256x256 images were used)
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        # Add normalization if you used it during training:
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    
    # Open image from bytes, convert to RGB, apply transforms, and add batch dimension
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = my_transforms(image).unsqueeze(0) 
    return tensor

def get_prediction(image_bytes, model):
    """
    Runs the inference on the PyTorch model and returns the class and confidence.
    """
    tensor = transform_image(image_bytes)
    
    # Use inference_mode for faster execution and less memory
    with torch.inference_mode():
        outputs = model(tensor)
    
    # Get probabilities and predicted class index
    probabilities = F.softmax(outputs, dim=1)
    _, predicted_idx = torch.max(outputs, 1)
    
    # Return human-readable result
    predicted_class = CLASSES[predicted_idx.item()]
    confidence = probabilities[0][predicted_idx].item()
    
    return predicted_class, confidence



from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn 
# The model architecture (ResNet9, conv_block) and utilities are imported from model_utils
# Note: These imports are crucial, even if PyTorch still complains about __main__.ResNet9
"""FastAPI backend (rewritten)

Entrypoint: run `uvicorn main:app --reload` from `fastapi-backend`.

This server uses `model_utils.load_model` to robustly load your .pth file
and exposes a POST `/predict` endpoint that accepts an image file and
returns the predicted plant disease class and confidence.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch

from model_utils import load_model, get_prediction, CLASSES


MODEL_PATH = Path(__file__).parent / 'models' / 'plant-disease-model-state.pth'


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model at startup and attach to app.state.model
    try:
        model = load_model(MODEL_PATH, device='cpu')
        app.state.model = model
        print('Model loaded successfully.')
    except Exception as e:
        # Fail fast: if model cannot be loaded, prevent the server from starting
        print(f'Error loading model during startup: {e}')
        raise RuntimeError(f'Failed to load model during startup: {e}')

    yield

    # Clean up
    if hasattr(app.state, 'model'):
        del app.state.model
        print('Model unloaded.')


app = FastAPI(title='Plant Disease Classifier API', lifespan=lifespan)

app.add_middlewfare(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your specific Frontend URL=["*"],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
def root():
    return {'message': 'Plant Disease Classifier API Online'}


@app.post('/api/diagnose')
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image.')

    if not hasattr(app.state, 'model'):
        raise HTTPException(status_code=503, detail='Model not loaded.')

    try:
        image_bytes = await file.read()
        # Get top-5 predictions (adjust k as you like)
        results = get_prediction(image_bytes, app.state.model, topk=5)

        top1 = results[0]
        # Convert confidences to percentages for the frontend display
        scores = [{'class': r['class'], 'confidence': round(r['confidence'] * 100, 2)} for r in results]

        return {
            'predicted_class': top1['class'],
            'confidence': round(top1['confidence'] * 100, 2),
            'scores': scores,
            'filename': file.filename,
        }
    except Exception as e:
        print(f'Prediction error: {e}')
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')



