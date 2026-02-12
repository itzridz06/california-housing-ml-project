from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import numpy as np
from model import predict_house

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

    data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                      Population, AveOccup, Latitude, Longitude]])

    prediction = predict_house(data)

    return templates.TemplateResponse("index.html",
                                      {"request": request,
                                       "prediction": round(prediction, 3)})
