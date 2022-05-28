from tkinter.tix import Select
from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model1 = pickle.load(open('LinearRegressionModel.pkl','rb'))
model2 = pickle.load(open('model.pkl', 'rb'))
model3=pickle.load(open('LinearRegressionModel3.pkl','rb'))
model4=pickle.load(open('LinearRegression4.pkl','rb'))
car=pd.read_csv('Cleaned Car.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/timeline')
def timeline():
    return render_template('timeline.html')

@app.route('/carbon_emission', methods=['GET','POST'])
def carbon():
    return render_template('carbon_emission.html')

@app.route('/car_price',methods=['GET','POST'])
def car_price():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    companies.insert(0,'Select Company')
    car_models.insert(0, 'Select Car Models')
    year.insert(0, 'Select Year')
    fuel_type.insert(0, 'Select Fuel Type')
    return render_template('car_price.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)

@app.route('/year',methods=['GET','POST'])
def year():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    fuel_type = sorted(car['fuel_type'].unique())
    companies.insert(0,'Select Company')
    car_models.insert(0, 'Select Car Models')
    fuel_type.insert(0, 'Select Fuel Type')
    return render_template('year.html',companies=companies, car_models=car_models, fuel_types=fuel_type)

@app.route('/km_driven',methods=['GET','POST'])
def car_km():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    companies.insert(0,'Select Company')
    car_models.insert(0, 'Select Car Models')
    year.insert(0, 'Select Year')
    fuel_type.insert(0, 'Select Fuel Type')
    return render_template('km_driven.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict1',methods=['POST'])
@cross_origin()
def predict1():
    company=request.form.get('company')
    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')
    prediction=model1.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)
    return str(np.round(prediction[0],2))


@app.route('/predict2',methods=['POST'])
def predict2():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model2.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('carbon_emission.html', prediction_text='CO2 Emission of the vehicle is : {}'.format(output))


@app.route('/predict3',methods=['POST'])
def predict3():
    company=request.form.get('company')
    car_model=request.form.get('car_models')
    fuel_type=request.form.get('fuel_type')
    prediction=model3.predict(pd.DataFrame(columns=['name', 'company',  'fuel_type'],
                              data=np.array([car_model,company,fuel_type]).reshape(1, 3)))
    print(prediction)
    return str(int(np.round(prediction[0])))


@app.route('/predict4',methods=['POST'])
def predict4():
    company=request.form.get('company')
    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    Price=request.form.get('Price')
    prediction=model4.predict(pd.DataFrame(columns=['name', 'company',  'fuel_type', 'year', 'Price'],
                              data=np.array([car_model,company,fuel_type, year,Price ]).reshape(1, 5)))
    print(prediction)
    return str(np.round(prediction[0]))

if __name__=='__main__':
    app.run()
