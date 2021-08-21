import os
import pickle
from sklearn.naive_bayes import GaussianNB

# function to process data and return it in correct format
def process_stars_data(data):
   
    processed = [
        {
            "Temperature": d.Temperature,
            "Relative_luminosity": d.Relative_luminosity,
            "Relative_radius": d.Relative_radius,
            "Absolute_magnitude": d.Absolute_magnitude,
            "Color": d.Color,
            "Spectral_class": d.Spectral_class,
            "Type": d.Type,
        }
        for d in data
    ]

    return processed
