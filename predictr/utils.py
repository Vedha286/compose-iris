import pickle
from sklearn.naive_bayes import GaussianNB


# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

clf_stars_score = GaussianNB()
clf = GaussianNB()

# function to train and load the model during startup
def init_models():
    init_stars_model()

def init_stars_model():
    global clf_stars_score
    clf_stars_score = pickle.load(open(".models/stars_nb.pkl", "rb"))


# function to predict the flower using the model
def predict_stars(query_data):
    x = list(query_data.dict().values())
    prediction = clf_stars_score.predict([x])[0]
    return get_credit_scorce_class(prediction)

def get_credit_scorce_class(type_int):
    if type_string == 0.0:  
        return "Brown Dwarf"
    elif type_string == 1.0:
        return "Red Dwarf"
    elif type_string == 2.0:  
        return "White Dwarf"
    elif type_string == 3.0:  
        return "Main Sequence"
    elif type_string == 4:  
        return "Super Giant"
    elif type_string == 5.0:  
        return "Hyper giant"
    else:
        return "Error"