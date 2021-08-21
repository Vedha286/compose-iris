import uvicorn
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from utils import init_models, train_model, train_model_stars
from typing import List

PREDICTR_ENDPOINT = os.getenv("PREDICTR_ENDPOINT")

# defining the main app
app = FastAPI(title="trainr", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", init_models)

# class which is expected in the payload while training
class TrainIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    flower_class: str

class StarTrainIn(BaseModel):
  Temperature: float
  Relative_luminosity: float
  Relative_radius: float
  Absolute_magnitude: float
  Color: str
  Spectral_class: str
  Type: str

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/star/train", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
def train_star(data: List[StarTrainIn]):
    train_model_stars(data)
    # tell predictr to reload the model
    response = requests.post(f"{PREDICTR_ENDPOINT}/reload_model")
    return {"detail": "Training successful"}

@app.post("/iris-flower/train", status_code=200)
# Route to further train the model based on user input in form of feedback loop
# Payload: FeedbackIn containing the parameters and correct flower class
# Response: Dict with detail confirming success (200)
def train(data: List[TrainIn]):
    train_model(data)
    # tell predictr to reload the model
    response = requests.post(f"{PREDICTR_ENDPOINT}/reload_model")
    return {"detail": "Training successful"}

# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=7777, reload=True)


# [
#   {
#     "sepal_length": 5.1,
#     "sepal_width": 3.5,
#     "petal_length": 1.4,
#     "petal_width": 0.2,
#     "flower_class": "Iris Setosa"
#   }
# ]


# [
#   {
#     "Temperature": 3068,
#     "Relative_luminosity": 0.0024,
#     "Relative_radius": 0.17,
#     "Absolute_magnitude": 16.12,
#     "Color": "Red",
#     "Spectral_class": "M",
#     "Type": "Red Dwarf"
#   }
# ]