"""model_utils (rewritten)

This module defines the ResNet9 architecture (as used in your notebook), the
class list, preprocessing, prediction helper, and a robust `load_model` that
tries to load either a state_dict or a pickled full model while using
`torch.serialization.safe_globals` to allowlist the ResNet9 global when
unpickling.
"""

from pathlib import Path
import io
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# --- Model architecture ---
def conv_block(in_channels, out_channels, pool: bool = False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # Use adaptive pooling so the classifier receives a 512-d feature
        # vector regardless of input spatial size. This prevents errors like
        # "mat1 and mat2 shapes cannot be multiplied" when clients send
        # larger images than used during training.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
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


# --- Classes (must match training) ---
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
    'Tomato___healthy',
]


# --- Preprocessing and prediction ---
def transform_image(image_bytes: bytes) -> torch.Tensor:
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes: bytes, model: nn.Module, topk: int = 5):
    """Run inference and return top-k class scores sorted by confidence.

    Returns a list of dicts: [{'class': class_name, 'confidence': float}, ...]
    The first element is the top-1 prediction.
    """
    tensor = transform_image(image_bytes)
    with torch.inference_mode():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)

        topk = min(topk, probs.size(1))
        top_probs, top_idxs = torch.topk(probs, k=topk, dim=1)
        top_probs = top_probs[0].cpu().tolist()
        top_idxs = top_idxs[0].cpu().tolist()

        results = []
        for idx, p in zip(top_idxs, top_probs):
            results.append({
                'class': CLASSES[int(idx)],
                'confidence': float(p)
            })
        return results


# --- Robust loader ---
def load_model(path: str | Path, device: str = 'cpu') -> nn.Module:
    """Load a model from `path` in a few common formats.

    Tries, in order:
    1. weights-only load (state_dict) with safe_globals allowlist.
    2. full-object load (torch.load) with safe_globals allowlist.

    Raises a RuntimeError with helpful instructions if both fail.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    device = torch.device(device)

    # Prepare a couple of fallbacks: prefer safe_globals if available.
    safe_globals_ctx = getattr(torch.serialization, 'safe_globals', None)

    # Helper to run a loader inside safe_globals when available
    def _with_safe_globals(func):
        if safe_globals_ctx is None:
            return func()
        else:
            with safe_globals_ctx([ResNet9, conv_block]):
                return func()

    # 1) Try weights-only (state_dict-like) load
    try:
        def _load_state():
            return torch.load(path, map_location=device, weights_only=True)

        state = _with_safe_globals(_load_state)
        if isinstance(state, dict):
            # Allow dicts that contain a nested 'state_dict' key
            if 'state_dict' in state and isinstance(state['state_dict'], dict):
                state_dict = state['state_dict']
            else:
                state_dict = state

            model = ResNet9(in_channels=3, num_classes=len(CLASSES))
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model
    except Exception:
        # swallow and try full load next
        pass

    # 2) Try full-model load (unsafe if from untrusted source)
    try:
        def _load_full():
            return torch.load(path, map_location=device, weights_only=False)

        model_obj = _with_safe_globals(_load_full)
        if isinstance(model_obj, nn.Module):
            model_obj.to(device)
            model_obj.eval()
            return model_obj
    except Exception as full_e:
        # Report both reasons
        raise RuntimeError(
            "Failed to load model. Tried state_dict and full-object loads. "
            f"Last error: {full_e}\n"
            "If this persists, re-save the model from your training notebook using:\n"
            "    torch.save(model.state_dict(), 'models/plant-disease-model-state.pth')\n"
            "and update the server to load the state_dict into ResNet9."
        )
