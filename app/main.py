from fastapi import FastAPI, Request, UploadFile, HTTPException, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import pickle
import pandas as pd
import json
import os

#- Constants
UPLOAD_DIRECTORY = "data/uploaded_images"
MODEL_PATH = "app/models/model3.pkl"
CITY_BIN_RULES_PATH = "data/city_bins.json"

#- Initialize FastAPI
app = FastAPI()

#- Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

#- Load model and bin rules
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(CITY_BIN_RULES_PATH, 'r') as rules_file:
    city_bin_rules = json.load(rules_file)

#- Mount the static files directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

#- Set up the templates directory
templates = Jinja2Templates(directory="frontend")

#- Helper function for class mapping
def reverse_mapping_function(predicted_class_numeric):
    class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
    reverse_mapping = {i: name for i, name in enumerate(class_names)}
    return reverse_mapping.get(predicted_class_numeric, "Unknown")

#- Helper function to create a DataFrame with image path
def create_image_dataframe(image_path):
    """Creates a DataFrame with a single row containing the image path."""
    return pd.DataFrame([{"path": image_path}])

#- Route to serve the HTML front end
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#- Endpoint for garbage classification
@app.post("/predict-bin/")
async def predict_bin(city: str = Form(...), file: UploadFile = File(...), model: str = Form("model.pkl")):
    # -Validate model selection
    model_path = f"app/models/{model}"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Model not found.")

    #- Load the selected model
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    #- Check if city exists in rules
    if city not in city_bin_rules:
        raise HTTPException(status_code=400, detail="City not recognized in bin rules.")
    
    #- Save uploaded image to the upload directory
    image_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    #- Verify if the image was saved
    if not os.path.exists(image_path):
        raise HTTPException(status_code=500, detail="Failed to save the uploaded image.")
    
    #- Prepare data and make prediction
    image_data = create_image_dataframe(image_path)
    predicted_numeric = loaded_model.predict(image_data)[0]
    waste_type = reverse_mapping_function(predicted_numeric)
    
    #- Get bin type based on city
    bin_type = city_bin_rules[city].get(waste_type, "Unrecognized type of waste")
    
    #- Response
    response = {
        "predicted_waste_type": waste_type,
        "recommended_bin": bin_type,
        "city": city
    }
    return JSONResponse(content=response)
