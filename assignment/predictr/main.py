import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from utils import init_models, predict_stars

# defining the main app
app = FastAPI(title="predictr", docs_url="/")

class StarTrainIn(BaseModel):
  Temperature: float
  Relative_luminosity: float
  Relative_radius: float
  Absolute_magnitude: float
  Color: str
  Spectral_class: str
  
class StarTrainOut(BaseModel):
    Type: str

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/predict_star", response_model=StarTrainOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_star(query_data: StarTrainIn):
    output = {"Type": predict_stars(query_data)}
    return output


@app.post("/reload_models", status_code=200)
# Route to reload the model from file
def reload_model():
    init_models()
    output = {"detail": "Model successfully loaded"}
    return output


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=9999, reload=True)
