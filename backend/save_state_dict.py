"""save_state_dict.py

Load an existing full-model checkpoint and save only the state_dict.

Run from repository root:
    python fastapi-backend/save_state_dict.py

This script assumes `model_utils.py` contains the `ResNet9` and `conv_block`
definitions used to create the original model. It will attempt to allowlist
those globals during unpickling (PyTorch 2.6+ safe loading).
"""

import sys
from pathlib import Path
import torch

import model_utils


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / 'models' / 'plant-disease-model-complete.pth'
OUT_FILE = ROOT / 'models' / 'plant-disease-model-state.pth'


def main():
    if not MODEL_FILE.exists():
        print(f"Model file not found: {MODEL_FILE}")
        return 2

    # Ensure the model class can be found under common module names used when
    # pickling from notebooks or interactive sessions.
    sys.modules['__main__'] = model_utils
    sys.modules['__mp_main__'] = model_utils

    # Allowlist the model globals (ResNet9, conv_block) if the API exists.
    try:
        torch.serialization.add_safe_globals([model_utils.ResNet9, model_utils.conv_block])
        print('Added safe globals for ResNet9 and conv_block')
    except Exception:
        # Older torch may not provide this API; that's fine.
        pass

    print(f'Loading checkpoint {MODEL_FILE} (this may execute code from the checkpoint — only do this for trusted files)')
    ckpt = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)

    # Extract state_dict
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state = ckpt['state_dict']
        else:
            # It may already be a state_dict
            state = ckpt
    else:
        # If it's a model object
        try:
            state = ckpt.state_dict()
        except Exception as e:
            print('Loaded object is neither dict nor module with state_dict():', type(ckpt), e)
            return 3

    torch.save(state, OUT_FILE)
    print(f'Saved state_dict to {OUT_FILE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
