from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

######import pickle files

regression = pickle.load(open('models/regression.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))


##routes for homepage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method=='POST':
        longitude = float(request.form.get('longitude'))
        latitude = float(request.form.get('latitude'))
        housing_median_age = float(request.form.get('housing_median_age'))
        total_rooms = float(request.form.get('total_rooms'))
        total_bedrooms = float(request.form.get('total_bedrooms'))
        population = float(request.form.get('population'))
        households = float(request.form.get('households'))
        median_income = float(request.form.get('median_income'))
        new_values = scaler.transform([[longitude,latitude,housing_median_age
                                        ,total_rooms,total_bedrooms,population,households,
                                        median_income]])
        result =  regression.predict(new_values)
        return render_template('home.html',result=round(result[0],2))
        
    else:
        return render_template('home.html')
        
    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12000)
    app.run(debug=True)