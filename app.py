from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Create a FastAPI instance
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the list of allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your models from the models directory
model_dir = os.path.join(os.path.dirname(__file__), 'models')
logistic_model_path = os.path.join(model_dir, 'best_logistic_model.pkl')
rf_model_path = os.path.join(model_dir, 'best_rf_model.pkl')

try:
    logistic_model = joblib.load(logistic_model_path)
    rf_model = joblib.load(rf_model_path)
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error("Error loading models: %s", e)
    raise RuntimeError("Model loading failed")

# Define the input data model with type constraints
class PredictionInput(BaseModel):
    features: conlist(float, min_length=20, max_length=20)  # Adjust min/max items based on your model


@app.post('/predict/logistic')
async def predict_logistic(input: PredictionInput):
    logging.info("Received input for logistic prediction: %s", input)
    try:
        # Convert input features to numpy array and reshape if necessary
        features = np.array(input.features).reshape(1, -1)
        prediction = logistic_model.predict(features)
        prediction_proba = logistic_model.predict_proba(features)

        # Convert numpy objects to Python native types before returning
        return {
            "prediction": int(prediction[0]),  # Convert numpy.int64 to Python int
            "probabilities": prediction_proba.tolist()  # Convert numpy array to list
        }
    except Exception as e:
        logging.error("Error during logistic prediction: %s", e)
        raise HTTPException(status_code=500, detail="Prediction error")

@app.post('/predict/random_forest')
async def predict_rf(input: PredictionInput):
    logging.info("Received input for random forest prediction: %s", input)
    try:
        features = np.array(input.features).reshape(1, -1)
        prediction = rf_model.predict(features)
        prediction_proba = rf_model.predict_proba(features)

        # Convert numpy objects to Python native types before returning
        return {
            "prediction": int(prediction[0]),  # Convert numpy.int64 to Python int
            "probabilities": prediction_proba.tolist()  # Convert numpy array to list
        }
    except Exception as e:
        logging.error("Error during random forest prediction: %s", e)
        raise HTTPException(status_code=500, detail="Prediction error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
