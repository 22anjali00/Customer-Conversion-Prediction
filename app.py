from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# load the trained model and encoders
model = joblib.load("saved/model.pkl")
scaler = joblib.load("saved/scaler.pkl")
ohe = joblib.load("saved/ohe.pkl")
le_marital = joblib.load("saved/le_marital.pkl")
job_qual = joblib.load("saved/job_qual.pkl")
le_call = joblib.load("saved/call_type.pkl")

def output(input_df):
    

    input_df["marital"] = le_marital.transform(input_df["marital"])
    input_df["education_qual"] = job_qual.transform(input_df["education_qual"])
    input_df["call_type"] = le_call.transform(input_df["call_type"])
    input_df["mon"] = mon.transform(input_df["mon"])
    input_df["prev_outcome"] = prev_outcome.transform(input_df["prev_outcome"])
    input_encoded = ohe.transform(input_df[["job"]])
    input_encoded_df = pd.DataFrame(input_encoded, columns=["job_" + str(i) for i in range(input_encoded.shape[1])])
    input_df = pd.concat([input_df, input_encoded_df], axis=1)
    input_df = input_df.drop("job", axis=1)


    input_df_scaled = scaler.transform(input_df)

    # make a prediction using the model
    prediction = model.predict(input_df_scaled)[0]
    if prediction == 1:
        result = "Customer will convert."
    else:
        result = "Customer will not convert."


# index page with input form
@app.route("/", methods=["GET", "POST"])
def index():
    

        # transform the input data to the format expected by the model
       
        
        return render_template("index.html")

# result page to display the prediction
@app.route("/result",methods=["GET", "POST"])
def result():
    if request.method == "POST":
        age = int(request.form["age"])
        job = request.form["job"]
        marital = request.form["marital"]
        edu = request.form["edu"]
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
        ans = output(input_df)
    return render_template("result.html",result = ans)

if __name__ == '__main__':
    app.run(debug=True)
