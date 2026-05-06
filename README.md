# MNIST Finger-Scribe

A high-precision handwriting digit classifier using scikit-learn and FastAPI.

## Project Structure

- `model_train.py`: Trains the MLP model using scikit-learn and saves it to `model.pkl`.
- `app.py`: FastAPI server that provides a `/predict` endpoint.
- `requirements.txt`: Python package requirements.
- `src/App.tsx`: React frontend with a drawing canvas interface.

## How to Run

### Local Development

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   npm install
   ```
2. **Start the application**:
   ```bash
   npm run dev
   ```
   This will start both the Vite frontend (port 3000) and the FastAPI backend (port 8000), with proxying configured.

### Training the Model
The model trains automatically on first run if `model.pkl` is missing, but you can trigger it manually:
```bash
npm run train
```

## Deployment
The application is configured to start with:
`uvicorn app:app --host 0.0.0.0 --port $PORT`
