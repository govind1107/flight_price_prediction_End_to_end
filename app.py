import sys
import threading

from flask import Flask,request,jsonify,render_template
import sklearn
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('xgb.pkl','rb'))

import faulthandler;
faulthandler.enable()

sys.setrecursionlimit(2097152)    # adjust numbers
threading.stack_size(134217728)

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        featureDict = {}
        duration = int(request.form['Duration'])
        day = int(request.form['Day'])
        month = int(request.form['Month'])
        dep_hour = int(request.form['dep_hour'])
        dep_min = int(request.form['dep_min'])
        Arrival_Hour = int(request.form['Arrival_Hour'])
        Arrival_min = int(request.form['Arrival_min'])

        Airline_type = request.form['Airline_type']
        featureDict['Airline_Air Asia'] = featureDict['Airline_Air India'] = featureDict['Airline_GoAir'] = featureDict['Airline_IndiGo'] =featureDict['Airline_Jet Airways']=featureDict['Airline_SpiceJet']= featureDict['Airline_Vistara']=featureDict['Airline_Multiple carriers']=0
        featureDict[Airline_type] = 1

        source = request.form['Source']
        featureDict['Source_Banglore'] = featureDict['Source_Chennai'] = featureDict['Source_Delhi'] = featureDict['Source_Kolkata'] = featureDict['Source_Mumbai']=0
        featureDict[source] = 1

        destination = request.form['Destination']
        featureDict['Destination_Banglore'] = featureDict['Destination_Cochin'] = featureDict['Destination_Delhi'] = featureDict['Destination_Hyderabad'] = featureDict['Destination_Kolkata'] = featureDict['Destination_New Delhi'] = 0
        featureDict[destination] = 1

        total_stops = request.form['Total_Stops']
        featureDict['Total_Stops_1 stop'] = featureDict['Total_Stops_2 stops'] = featureDict['Total_Stops_3 stops'] = featureDict['Total_Stops_4 stops'] = featureDict['Total_Stops_non-stop'] = 0
        featureDict[total_stops] = 1

        additional_Info = request.form['Additional_Info']
        featureDict['Additional_Info_1 Long layover'] =featureDict['Additional_Info_1 Short layover'] =featureDict['Additional_Info_2 Long layover'] =featureDict['Additional_Info_Business class'] =featureDict['Additional_Info_Change airports'] = featureDict['Additional_Info_In-flight meal not included'] = featureDict['Additional_Info_No Info'] = featureDict['Additional_Info_No check-in baggage included'] = featureDict['Additional_Info_Red-eye flight'] = 0
        featureDict[additional_Info] = 1

        featurelist = [duration, day, month, dep_hour, dep_min, Arrival_Hour, Arrival_min,featureDict['Airline_Air Asia'] , featureDict['Airline_Air India'] , featureDict['Airline_GoAir'] , featureDict['Airline_IndiGo'] ,featureDict['Airline_Jet Airways'],featureDict['Airline_SpiceJet'],featureDict['Airline_Vistara'],featureDict['Airline_Multiple carriers'],featureDict['Source_Banglore'] , featureDict['Source_Chennai'] , featureDict['Source_Delhi'] , featureDict['Source_Kolkata'] , featureDict['Source_Mumbai'],featureDict['Destination_Banglore'] , featureDict['Destination_Cochin'] , featureDict['Destination_Delhi'] , featureDict['Destination_Hyderabad'] , featureDict['Destination_Kolkata'] , featureDict['Destination_New Delhi'],featureDict['Total_Stops_1 stop'] , featureDict['Total_Stops_2 stops'] , featureDict['Total_Stops_3 stops'] , featureDict['Total_Stops_4 stops'] , featureDict['Total_Stops_non-stop'],featureDict['Additional_Info_1 Long layover'] ,featureDict['Additional_Info_1 Short layover'] ,featureDict['Additional_Info_2 Long layover'] ,featureDict['Additional_Info_Business class'] ,featureDict['Additional_Info_Change airports'] ,featureDict['Additional_Info_In-flight meal not included'] , featureDict['Additional_Info_No Info'] , featureDict['Additional_Info_No check-in baggage included'] , featureDict['Additional_Info_Red-eye flight']]
        print(featurelist)

        featureList = np.array(featurelist)
        featureList = featureList.reshape((1, len(featureList)))
        prediction = model.predict(featureList)
        output = round(prediction[0],2)

        print(len(featurelist))
        return render_template('index.html', prediction_text="price of ticket is  {}".format(output))
    else:
        return render_template('index.html')







if __name__ == '__main__':
    print('app is running')
    app.run()