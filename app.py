import numpy as np
import pandas as pd
import joblib
import flask
# print(flask.__version__)
from flask import Flask,request
from house_function import first_letter
pipeline=joblib.load("titan.model")
# print(pipeline)

# demarrage de mon application
app=Flask('__name__')
# page d'accueil
@app.route("/")
def index():
  return "<h1>voici ma page de prediction pour le titanic<h1>"

  # je crée la deuxième page
  # GET c'est pour envoyer des questions
@app.route('/ping',methods=['GET'])
def ping():
  return("pong",200)

  # predicte

@app.route('/predict',methods=['POST'])
def predict():
  df=pd.DataFrame(request.json)  
  resultat=pipeline.predict(df)[0]
  return (str(resultat),200)
# le transfert de ladata dans le web se fait sous forme de json
  
  # obligation pour demarer ma page
if __name__ == "__main__":
  app.run(host="0.0.0.0")  