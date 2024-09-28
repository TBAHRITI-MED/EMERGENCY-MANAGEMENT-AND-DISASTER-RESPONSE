from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from keras.models import load_model
import cv2
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load EmergencyNet model
model = load_model('model_emergencyNet.h5')

# Define function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Assuming model input size is 224x224
    img = img / 127.5 - 1  # Normalize
    return np.expand_dims(img, axis=0)

# Define endpoint to predict emergency from an image
@app.post("/predict/")
async def predict_emergency(file: UploadFile = File(...)):
    try:
        image = preprocess_image(file.file)
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
        classes = ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']
        result = {'class': classes[class_index], 'confidence': float(prediction[0][class_index])}
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={'error': str(e)})

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
