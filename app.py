from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib
import os
import  numpy as np
import pickle
from tensorflow.keras.models import load_model

app= Flask(__name__)
app.secret_key = "f06464a7cad8df443c9af33463cf6138"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result",methods=['POST','GET'])
def result():
     # Store form data in session
    session['gender'] = int(request.form['gender'])
    session['age'] = int(request.form['age'])
    session['hypertension'] = int(request.form['hypertension'])
    session['heart_disease'] = int(request.form['heart_disease'])
    session['ever_married'] = int(request.form['ever_married'])
    session['work_type'] = int(request.form['work_type'])
    session['Residence_type'] = int(request.form['Residence_type'])
    session['avg_glucose_level'] = float(request.form['avg_glucose_level'])
    session['bmi'] = float(request.form['bmi'])
    session['smoking_status'] = int(request.form['smoking_status'])

    # Prepare data for prediction
    x = np.array([session['gender'], session['age'], session['hypertension'], session['heart_disease'], 
                  session['ever_married'], session['work_type'], session['Residence_type'],
                  session['avg_glucose_level'], session['bmi'], session['smoking_status']]).reshape(1, -1)

    scaler_path=os.path.join('C:/Users/VEDHA/OneDrive/Documents/Brain_Stroke_Prediction-main/Model/Code','models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    #Decision Tree
    # model_path=os.path.join('C:/Users/ASUS/OneDrive/Model/Code','models/dt.sav')
    # dt=joblib.load(model_path)
    # Y_pred=dt.predict(x)

    # Final Model
    # model_path=os.path.join('C:/Users/ASUS/OneDrive/Model/Code','models/finalized_model.sav')
    # final=joblib.load(model_path)
    # Y_pred=final.predict(x)


    #Final
    model_path=os.path.join('C:/Users/VEDHA/OneDrive/Documents/Brain_Stroke_Prediction-main/Model/Code','models/rf.sav')
    rf=joblib.load(model_path)
    Y_pred=rf.predict(x)

    #Displaying the prediction
    print("Stroke Prediction: ",Y_pred)

    # for No Stroke Risk
#     if Y_pred==1:
#         return render_template('stroke.html')
#     else:
#         return render_template('nostroke.html')


# if __name__=="__main__":
#     app.run(debug=True,port=7384)

 # If no stroke risk, render nostroke.html

    # If Y_pred is 1 (stroke risk), render the stroke type prediction form
    if Y_pred==1:
        return render_template('stroke_type_result.html')

    else:
        return render_template('nostroke.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Load model and feature names
    model = joblib.load(r'C:\Users\VEDHA\OneDrive\Documents\Brain_Stroke_Prediction-main\Model\Code\model2\stroke_type_model.pkl')
    features = joblib.load(r'C:\Users\VEDHA\OneDrive\Documents\Brain_Stroke_Prediction-main\Model\Code\model2\stroke_symptom_features.pkl')
    risk_model = load_model(r'C:\Users\VEDHA\OneDrive\Documents\Brain_Stroke_Prediction-main\Model\Code\model3\stroke_risk_model.h5')
    scaler = joblib.load(r'C:\Users\VEDHA\OneDrive\Documents\Brain_Stroke_Prediction-main\Model\Code\model3\scaler.save')

    #get data from form
    data = request.get_json()
    symptoms = data['symptoms']  # list of 10 binary values
    bp = data.get('bp')          # numerical BP value
    stroke_history = data.get('stroke_history')
    family_history = data.get('family_history')
   
    # Append BP to the symptom list
    input_features = symptoms 
    input_array = [input_features]  # shape: [[...]] for model input

    prediction = model.predict(input_array)[0]
    print("Type prediction: ", prediction)
    label = "Ischemic" if prediction == 1 else "Hemorrhagic"
    
    # Create main input
    input_data = np.array([[session['age'], session['hypertension'], session['heart_disease'], session['avg_glucose_level'], session['bmi'], stroke_history, family_history]])

    # Create engineered features
    age_glucose = input_data[0][0] * input_data[0][3]  # age * avg_glucose_level
    bmi_hypertension = input_data[0][4] * input_data[0][1]  # bmi * hypertension

    # Final input with engineered features
    final_input = np.hstack([input_data, [[age_glucose, bmi_hypertension]]])

    # Scale 
    scaled_input = scaler.transform(final_input)
    
    #predict
    risk_prediction = risk_model.predict(scaled_input)
    predicted_risk_class = np.argmax(risk_prediction)

    # Risk label
    risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
    session['type'] = label
    session['risk'] = risk_mapping[predicted_risk_class]
    print(f"Risk prediction:  {risk_mapping[predicted_risk_class]}")
    return jsonify({'prediction': label, 'risk': risk_mapping[predicted_risk_class]})



@app.route('/stroke_type_result')
def stroke_type_result():
    stroke_type = request.args.get('type')
    stroke_risk = request.args.get('risk')
    return render_template('stroke.html', stroke_type=stroke_type,stroke_risk=stroke_risk)


if __name__ == "__main__":
    app.run(debug=True, port=7384)