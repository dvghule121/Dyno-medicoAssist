from flask import Flask, request, render_template
from train import Dataset as ds

app = Flask(__name__)


@app.route("/about", methods=["GET", "POST"])
def about():
    return render_template("about.html")

@app.route("/", methods=["GET", "POST"])
def predict_disease():
    if request.method == "POST":
        symptoms = request.form["inp"].lower()
        syms = []
        if len(symptoms) != 0:
            user_input = symptoms
            dm = ds()
            disease = dm.predict_tag(user_input)
            treatment = []
            if disease != 0:
                for i in disease:
                    syms.append(dm.return_symp(i))
                    treatment.append(dm.treat(i))
            elif disease == 0:
                disease = ["Oops!! Disease Not Found "]
                syms = ["Looks like symptoms are different than our dataset"]
                treatment = ["Null"]


    try:
        return render_template("res.html", dises=disease, treatment=treatment, symptoms=syms)
    except UnboundLocalError:
        return render_template("res.html", disease="predicted disease will be shown here", treatment="treatment",
                               symptoms="symptoms")


if __name__ == '__main__':
    app.run(debug=True)
