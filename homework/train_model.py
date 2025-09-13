"""Build, deploy and access a model using scikit-learn"""

import pickle

import pandas as pd 
from sklearn.linear_model import LinearRegression 

df = pd.read_csv("files/input/house_data.csv", sep=",")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]]

estimator = LinearRegression()
estimator.fit(features, target)

with open("homework/house_predictor.pkl", "wb") as file: #El archivo es binario .pkl, se encarga del servidor
    pickle.dump(estimator, file) #guardar el modelo en el disco