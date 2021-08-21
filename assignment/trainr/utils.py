import os
import pickle
from sklearn.naive_bayes import GaussianNB
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import List
from pydantic import BaseModel
from sklearn.metrics import accuracy_score


class StarTrainIn(BaseModel):
  Temperature: float
  Relative_luminosity: float
  Relative_radius: float
  Absolute_magnitude: float
  Color: str
  Spectral_class: str
  Type: str


class StarTrainModify:
    def __init__(self, Temperature, Relative_luminosity, Relative_radius, Absolute_magnitude, Color, Spectral_class, Type):
        self.Temperature = Temperature
        self.Relative_luminosity = Relative_luminosity
        self.Relative_radius = Relative_radius
        self.Absolute_magnitude = Absolute_magnitude
        self.Color = Color
        self.Spectral_class = Spectral_class
        self.Type = Type

    def values(self):
        return [self.Temperature, self.Relative_luminosity, self.Relative_radius, self.Absolute_magnitude, self.Color, self.Spectral_class]

stars_model_file = "models/stars_nb.pkl"

# function to train and load the model during startup
def init_models():
    init_stars_model()

def init_stars_model():
    
    if not os.path.isfile(stars_model_file):
        clf = GaussianNB()
        pickle.dump(clf, open(stars_model_file, "wb"))
        
        data = download_data()
        X_train, X_test, y_train, y_test = prepare_data(data.itertuples(index=True))

        
        clf.fit(X_train, y_train)
        if len(data) > 10:
            accuracy = accuracy_score(y_test, clf.predict(X_test))
            print(f"Accuracy: " + str(accuracy))


        pickle.dump(clf, open(stars_model_file, "wb"))
        print("initilized and saved stars model")
    

def download_data():
      global DATA #Use x from the global space
      DATA = read_csv("data/stars.csv", sep=";", header=0, names=["Temperature", "Relative_luminosity", "Relative_radius", 
      "Absolute_magnitude", "Color", "Spectral_class", "Type"])
       
      return pd.DataFrame(DATA)

# function to prepare the data
def prepare_data(data):

    data = substituteStars(data)
    
    X = [d.values() for d in data]
    y = [d.Type for d in data]

    if len(data) > 10:
        X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=0.3)
        return X_train, X_test, y_train, y_test
    else: 
        return X, [], y, []

# function to train and save the model as part of the feedback loop
def train_model_stars(data: List[StarTrainIn]):
    # load the model
    clf = pickle.load(open(stars_model_file, "rb"))
    X_train, X_test, y_train, y_test = prepare_data(data)
 
    clf.fit(X_train, y_train)
    if len(data) > 10:
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        print(f"Accuracy: " + str(accuracy))

    # save the model
    print("done")
    pickle.dump(clf, open(stars_model_file, "wb"))
    return

def substituteStars(df):
    newData = []
    for d in df:
        # print(d)
        type_string = d.Type.strip().replace(" ", "").replace("-", "").lower()
        type_int = -1
        if type_string == "browndwarf":  
            type_int = 0.0
        elif type_string == "reddwarf":  
            type_int = 1.0
        elif type_string == "red dwarf":  
            type_int = 1.0
        elif type_string == "whitedwarf":  
            type_int = 2.0
        elif type_string == "mainsequence":  
            type_int = 3.0
        elif type_string == "supergiant":  
            type_int = 4.0
        elif type_string == "hypergiant":  
            type_int = 5.0
        elif type_string == "hypergiants":  
            type_int = 5.0
        elif type_string == "supergiants":  
            type_int = 4.0
        else:
            # print("Type:" + type_string + "-")
            type_int = -2


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
        
        
        newD = StarTrainModify(d.Temperature, d.Relative_luminosity, d.Relative_radius, d.Absolute_magnitude, colour_int, spectral_class, type_int)
        newData.append(newD)
    
    return newData
