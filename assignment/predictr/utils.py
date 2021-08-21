import pickle
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
from typing import List


class StarTrainIn(BaseModel):
  Temperature: float
  Relative_luminosity: float
  Relative_radius: float
  Absolute_magnitude: float
  Color: str
  Spectral_class: str


class StarTrainModify:
    def __init__(self, Temperature, Relative_luminosity, Relative_radius, Absolute_magnitude, Color, Spectral_class):
        self.Temperature = Temperature
        self.Relative_luminosity = Relative_luminosity
        self.Relative_radius = Relative_radius
        self.Absolute_magnitude = Absolute_magnitude
        self.Color = Color
        self.Spectral_class = Spectral_class

    def values(self):
        return [self.Temperature, self.Relative_luminosity, self.Relative_radius, self.Absolute_magnitude, self.Color, self.Spectral_class]


clf_stars_score = GaussianNB()
clf = GaussianNB()
stars_model_file = "models/stars_nb.pkl"

# function to train and load the model during startup
def init_models():
    init_stars_model()

def init_stars_model():
    global clf_stars_score
    clf_stars_score = pickle.load(open(stars_model_file, "rb"))

# function to predict the flower using the model
def predict_stars(query_data):
    data =substituteStars(query_data)
    # x = list(query_data.dict().values())
    x = data.values()
    print(x)
    print()
    print()
    prediction = clf_stars_score.predict([x])[0]
    star_type = get_star_class(prediction)
    print(star_type)
    return star_type

def get_star_class(type_int):
    if type_int == 0.0:  
        return "Brown Dwarf"
    elif type_int == 1.0:
        return "Red Dwarf"
    elif type_int == 2.0:  
        return "White Dwarf"
    elif type_int == 3.0:  
        return "Main Sequence"
    elif type_int == 4:  
        return "Super Giant"
    elif type_int == 5.0:  
        return "Hyper giant"
    else:
        return "Error"

def substituteStars(d: StarTrainIn):
    colour_int=-1
    color_string = d.Color.strip().replace(" ", "").replace("-", "").lower()
    if color_string == "white":  
        colour_int = 0.0
    elif color_string == "red":  
        colour_int = 1.0
    elif color_string == "blue":  
        colour_int = 2.0
    elif color_string == "yellow":  
        colour_int = 3.0
    elif color_string == "yelloworange":  
        colour_int = 4.0
    elif color_string == "bluewhite":  
        colour_int = 5.0
    elif color_string == "orange":  
        colour_int = 6.0
    elif color_string == "yellowish":  
        colour_int = 7.0
    elif color_string == "yellowwhite":  
        colour_int = 8.0
    elif color_string == "whiteyellow":  
        colour_int = 9.0
    elif color_string == "whitish":
        colour_int = 10.0
    elif color_string == "yellowishwhite":  
        colour_int = 11.0
    elif color_string == "yellowishwhite":  
        colour_int = 12.0
    elif color_string == "paleyelloworange":  
        colour_int = 13.0
    elif color_string == "orangered":  
        colour_int = 14.0
    else:
        # print("Color: " + color_string)
        colour_int = -2

    spectral_class_string = d.Spectral_class.strip().replace(" ", "").replace("-", "").upper()
    spectral_class = -1
    if spectral_class_string == "O":
        spectral_class = 0.0
    elif spectral_class_string == "B":
        spectral_class = 1.0
    elif spectral_class_string == "A":
        spectral_class = 2.0
    elif spectral_class_string == "F":
        spectral_class = 3.0
    elif spectral_class_string == "G":
        spectral_class = 4.0
    elif spectral_class_string == "K":
        spectral_class = 5.0
    elif spectral_class_string == "M":
        spectral_class = 6.0
    else:
        spectral_class= -1
        # print("Spectral_class: " + spectral_class_string)
    
    
    newD = StarTrainModify(d.Temperature, d.Relative_luminosity, d.Relative_radius, d.Absolute_magnitude, colour_int, spectral_class)
    return newD