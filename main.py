from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import numpy as np
from model import predict_house

# Initialize FastAPI app
app = FastAPI()

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")


# ===============================
# Home Route
# ===============================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ===============================
# Prediction Route
# ===============================
@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            MedInc: float = Form(...),
            HouseAge: float = Form(...),
            AveRooms: float = Form(...),
            AveBedrms: float = Form(...),
            Population: float = Form(...),
            AveOccup: float = Form(...),
            Latitude: float = Form(...),
            Longitude: float = Form(...)):

    # Prepare input array
    data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                      Population, AveOccup, Latitude, Longitude]])

    # Get prediction (in 100k dollars)
    prediction = predict_house(data)

    # Convert to actual dollar value
    prediction_dollar = prediction * 100000

    # Format with commas and 2 decimal places
    formatted_prediction = f"${prediction_dollar:,.2f}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": formatted_prediction
        }
    )
