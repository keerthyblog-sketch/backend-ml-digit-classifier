import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import threading

app = FastAPI(title="MNIST Digit Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_status = "loading"  # loading | ready | error

def load_or_train_model():
    global model, model_status
    model_path = "model.pkl"
    try:
        if os.path.exists(model_path):
            print("Loading existing model...")
            model = joblib.load(model_path)
        else:
            print("Training model (this takes a few minutes)...")
            # Inline fast training to avoid subprocess issues
            from sklearn.datasets import fetch_openml
            from sklearn.neural_network import MLPClassifier
            from sklearn.model_selection import train_test_split

            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
            X, y = mnist["data"] / 255.0, mnist["target"]
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)

            clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20,
                                solver="sgd", learning_rate_init=0.1, random_state=1)
            clf.fit(X_train, y_train)
            joblib.dump(clf, model_path)
            model = clf

        model_status = "ready"
        print("Model ready.")
    except Exception as e:
        model_status = "error"
        print(f"Model loading/training failed: {e}")

@app.on_event("startup")
async def startup_event():
    # Run in background so Render's health check passes immediately
    t = threading.Thread(target=load_or_train_model, daemon=True)
    t.start()

class PredictRequest(BaseModel):
    image: List[float]

@app.get("/")
async def root():
    return {"message": "MNIST Classifier API is running", "model_status": model_status}

@app.post("/predict")
async def predict(data: PredictRequest):
    if model_status == "loading":
        raise HTTPException(status_code=503, detail="Model is still loading, please retry in a moment.")
    if model_status == "error":
        raise HTTPException(status_code=500, detail="Model failed to load.")

    try:
        img_array = np.array(data.image).reshape(1, -1)
        if np.max(img_array) > 1.0:
            img_array = img_array / 255.0

        prediction = model.predict(img_array)[0]

        try:
            probabilities = model.predict_proba(img_array)[0]
            confidence = float(np.max(probabilities))
        except Exception:
            confidence = 1.0

        return {"prediction": str(prediction), "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
