from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Global variable to store the model
model = None

def load_model():
    """Load the model with proper error handling"""
    global model
    try:
        model_path = 'model/credit_risk_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded successfully")
            return True
        else:
            print(f"Model file not found at {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            if os.path.exists('model'):
                print(f"Files in model directory: {os.listdir('model')}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Try to load the model when the app starts
model_loaded = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return render_template('index.html', 
                             prediction=None, 
                             proba=None, 
                             error="Model not available. Please check server logs.")
    
    data = request.form

    try:
        person_income = float(data['person_income'])
        loan_amnt = float(data['loan_amnt'])
        loan_percent_income = loan_amnt / person_income if person_income != 0 else 0

        input_data = [[
            int(data['person_age']),
            person_income,
            float(data['person_emp_length']),
            data['loan_grade'],
            data['person_home_ownership'],
            data['loan_intent'],
            loan_amnt,
            float(data['loan_int_rate']),
            loan_percent_income,
            int(data['cb_person_default_on_file']),
            int(data['cb_person_cred_hist_length'])
        ]]

        cols = ['person_age', 'person_income', 'person_emp_length', 'loan_grade',
                'person_home_ownership', 'loan_intent', 'loan_amnt', 'loan_int_rate',
                'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']

        input_df = pd.DataFrame(input_data, columns=cols)

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        return render_template('index.html', prediction=pred, proba=round(proba, 3))

    except Exception as e:
        return render_template('index.html', 
                             prediction=None, 
                             proba=None, 
                             error=f"Prediction error: {str(e)}")

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'working_directory': os.getcwd()
    }
    return status

if __name__ == '__main__':
    app.run(debug=True)