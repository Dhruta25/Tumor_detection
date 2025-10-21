from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Brain Tumor Detection API")

# Load the trained model
MODEL_PATH = "/Users/dhrutamacm2/Desktop/tumor_detection_Dl/model.h5"
model = load_model(MODEL_PATH, compile = False)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Mount the uploads folder to serve static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# Pydantic model for response
class PredictionResponse(BaseModel):
    result: str
    confidence: str
    file_path: str


# Helper function to predict tumor type
def predict_tumor(image_path: str):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        result = "No Tumor"
    else:
        result = f"Tumor: {class_labels[predicted_class_index]}"

    return result, confidence_score


@app.get("/", response_class=HTMLResponse)
def home():
    """
    Simple HTML upload form for testing.
    """
    return """
    <html>
        <body style="text-align:center; font-family:sans-serif;">
            <h2>Brain Tumor Detection</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and predict tumor type.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Save uploaded image
    file_path = UPLOAD_FOLDER / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Make prediction
    result, confidence = predict_tumor(str(file_path))

    response = PredictionResponse(
        result=result,
        confidence=f"{confidence * 100:.2f}%",
        file_path=f"/uploads/{file.filename}"
    )

    return response


@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """
    Serve uploaded image file.
    """
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)