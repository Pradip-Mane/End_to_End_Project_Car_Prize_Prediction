from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)


## Route for a home page

@application.route('/')
def index():
    return render_template('index.html') 

@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            symboling=request.form.get('symboling'),
            fueltype=request.form.get('fueltype'),
            aspiration=request.form.get('aspiration'),
            doornumber=request.form.get('doornumber'),
            carbody=request.form.get('carbody'),
            drivewheel=request.form.get('drivewheel'),
            enginelocation=request.form.get('enginelocation'),
            wheelbase=float(request.form.get('wheelbase')),
            carlength=float(request.form.get('carlength')),
            carheight=float(request.form.get('carheight')),
            carwidth=float(request.form.get('carwidth')),
            curbweight=request.form.get('curbweight'),
            enginetype=request.form.get('enginetype'),
            cylindernumber=request.form.get('cylindernumber'),
            enginesize=request.form.get('enginesize'),
            fuelsystem=request.form.get('fuelsystem'),
            boreratio=float(request.form.get('boreratio')),
            stroke=float(request.form.get('stroke')),
            compressionratio=float(request.form.get('compressionratio')),
            horsepower=request.form.get('horsepower'),
            peakrpm=request.form.get('peakrpm'),
            citympg=request.form.get('citympg'),
            highwaympg=request.form.get('highwaympg')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    application.run(host="0.0.0.0", debug=True) 