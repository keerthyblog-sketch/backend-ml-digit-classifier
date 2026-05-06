import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import subprocess

app = FastAPI(title="MNIST Digit Classifier")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

def load_or_train_model():
    global model
    model_path = 'model.pkl'
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("Model not found. Training model automatically...")
        # We can run the training script or implement a fast training here
        # For simplicity and to match requirements, we'll run the training script
        try:
            subprocess.run(["python3", "model_train.py"], check=True)
            model = joblib.load(model_path)
        except Exception as e:
            print(f"Failed to train model: {e}")
            raise RuntimeError("Model training failed")

@app.on_event("startup")
async def startup_event():
    load_or_train_model()

class PredictRequest(BaseModel):
    # Expecting a list of 784 normalized pixel values (0-1)
    # or a list of integers (0-255)
    image: List[float]

@app.get("/")
async def root():
    return {"message": "MNIST Classifier API is running"}

@app.post("/predict")
async def predict(data: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert list to numpy array and reshape for prediction
        img_array = np.array(data.image).reshape(1, -1)
        
        # Ensure normalization if user sent 0-255
        if np.max(img_array) > 1.0:
            img_array = img_array / 255.0
            
        prediction = model.predict(img_array)[0]
        
        # Get confidence scores (probabilities)
        try:
            probabilities = model.predict_proba(img_array)[0]
            confidence = float(np.max(probabilities))
        except:
            # Some models don't support predict_proba
            confidence = 1.0
            
        return {
            "prediction": str(prediction),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# If you want to serve the frontend, you'd add this:
# from fastapi.staticfiles import StaticFiles
# if os.path.exists("dist"):
#     app.mount("/", StaticFiles(directory="dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
