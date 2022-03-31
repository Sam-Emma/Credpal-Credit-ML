# -*- coding: utf-8 -*-

from flask import Flask, request,  render_template
import model
app = Flask(__name__)
@app.route("/", methods =["POST","GET"])
def Home():
    predict_real = ""
    confid = ""
    salary = ""
    education = ""
    avg_pay_delay = ""
    if request.method == "POST":
        salary = request.form["salary"]
        education = request.form["education"]
        avg_pay_delay = request.form["avg_pay_delay"]
        predict_real, confid = model.predict(salary, education, avg_pay_delay)

    return render_template("index.html",target = predict_real, confidence = confid, salary_input =salary,education_input = education, apd_input = avg_pay_delay)


if __name__ == "__main__":
    app.run(debug = True,use_reloader=False)
