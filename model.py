from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'supersecretkey'



# Load the models
model = joblib.load("saved/model.pkl")
scaler = joblib.load("saved/scaler.pkl")
ohe = joblib.load("saved/ohe.pkl")
le_marital = joblib.load("saved/le_marital.pkl")
job_qual = joblib.load("saved/job_qual.pkl")
le_call = joblib.load("saved/call_type.pkl")
mon = joblib.load("saved/mon.pkl")
prev_outcome = joblib.load("saved/prev_outcome.pkl")



# Define a function to make predictions
def make_predictions(input_df):
    input_df["marital"] = le_marital.transform(input_df["marital"])
    input_df["education_qual"] = job_qual.transform(input_df["education_qual"])
    input_df["call_type"] = le_call.transform(input_df["call_type"])
    input_df["mon"] = mon.transform(input_df["mon"])
    input_df["prev_outcome"] = prev_outcome.transform(input_df["prev_outcome"])
    input_encoded = ohe.transform(input_df[["job"]])
    input_encoded_df = pd.DataFrame(input_encoded, columns=["job_" + str(i) for i in range(input_encoded.shape[1])])
    input_df = pd.concat([input_df, input_encoded_df], axis=1)
    input_df = input_df.drop("job", axis=1)

    
    

    prediction =  model.predict(input_df)
    if prediction == 1:
        result = "Customer will convert."
    else:
        result = "Customer will not convert."
    
    return result

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/result', methods=['POST'])
def predict():
    
    age = int(request.form["age"])
    job = request.form["job"]
    marital = request.form["marital"]
    edu = request.form["education_qual"]
    call_type = request.form["call_type"]
    day = int(request.form["day"])
    mon = request.form["mon"]
    dur = int(request.form["dur"])
    num_calls = int(request.form["num_calls"])
    prev_outcome = request.form["prev_outcome"]
    input_df = pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education_qual": [edu],
        "call_type": [call_type],
        "day": [day],
        "mon": [mon],
        "dur": [dur],
        "num_calls": [num_calls],
        "prev_outcome": [prev_outcome]
    })
    ans = make_predictions(input_df)
    print(ans[0])

    # Render the results template with the prediction
    return render_template('result.html', prediction=ans)



# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
