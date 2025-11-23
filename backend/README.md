# FastAPI backend for Plant Disease Classifier

This backend exposes a small FastAPI app that loads your PyTorch model and
provides a `/predict` endpoint to classify plant disease images.

How to run

1. Create a virtual environment and install dependencies (macOS / zsh):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start the server:

```bash
cd fastapi-backend
uvicorn main:app --reload
```

Notes about model formats

- Preferred: save the model weights only in your training notebook:

```python
torch.save(model.state_dict(), 'models/plant-disease-model-state.pth')
```

Then update `MODEL_PATH` in `main.py` if you use a different filename.

- The server attempts to load several common formats (state_dict or a full
  pickled model). If loading fails with a message about `ResNet9` or unsafe
  globals, re-save the weights as a `state_dict` as shown above.

Endpoints

- GET / -> health
- POST /predict -> multipart form with `file` (image) field. Returns JSON with
  `predicted_class` and `confidence` (percentage).
