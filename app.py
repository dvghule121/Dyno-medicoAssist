from flask import Flask, request, render_template
from train import Dataset as ds

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict_disease():
    disease = "predicted disease will be shown here"
    treatment = "treatment"
    symptoms = "symptoms"
    if request.method == "POST":
        symptoms = request.form["inp"].lower()
        if len(symptoms) != 0:
            user_input = symptoms
            dm = ds()
            disease = dm.predict_tag(user_input)
            treatment = []
            for i in disease:
                treatment.append(dm.treat(i))

    try:
        return render_template("res.html", dises=disease, treatment=treatment, symptoms=symptoms)
    except UnboundLocalError:
        return render_template("res.html", disease="predicted disease will be shown here", treatment="treatment",
                               symptoms="symptoms")


if __name__ == '__main__':
    app.run(debug=True)
