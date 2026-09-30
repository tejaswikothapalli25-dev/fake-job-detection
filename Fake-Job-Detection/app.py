from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib

app = Flask(__name__)

# Secret key for login session
app.secret_key = "fake-job-detection-secret-key"

# Load trained ML model and TF-IDF vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")


# ---------------------------------------
# FAKE JOB PREDICTION FUNCTION
# ---------------------------------------

def predict_job_posting(job_description, job_url=""):

    # Combine job description and URL
    text = f"{job_description} {job_url}"

    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(text_tfidf)[0]

    # Calculate confidence
    confidence = max(probabilities) * 100

    # Prediction result
    if prediction == 1:
        result = "Fake"
        status = "Fraudulent"
    else:
        result = "Genuine"
        status = "Safe"

    return {
        "prediction": result,
        "confidence": round(confidence, 2),
        "status": status
    }


# ---------------------------------------
# LOGIN PAGE
# ---------------------------------------

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        # Demo student login
        if username == "student" and password == "12345":

            session["logged_in"] = True
            session["username"] = username

            return redirect(url_for("dashboard"))

        else:

            return render_template(
                "login.html",
                error="Invalid username or password"
            )

    return render_template("login.html")


# ---------------------------------------
# DASHBOARD
# ---------------------------------------

@app.route("/dashboard")
def dashboard():

    # Check login
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        username=session.get("username")
    )


# ---------------------------------------
# FAKE JOB DETECTION PAGE
# ---------------------------------------

@app.route("/detect")
def detect():

    # Check login
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("index.html")


# ---------------------------------------
# ANALYZE JOB API
# ---------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():

    # Check login
    if not session.get("logged_in"):
        return jsonify({
            "error": "Please login first."
        }), 401

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Invalid request."
            }), 400

        job_description = data.get(
            "job_description",
            ""
        )

        job_url = data.get(
            "job_url",
            ""
        )

        # Check empty input
        if not job_description and not job_url:

            return jsonify({
                "error": "Please enter a job description."
            }), 400

        # Predict fake or genuine
        result = predict_job_posting(
            job_description,
            job_url
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------
# LOGOUT
# ---------------------------------------

@app.route("/logout")
def logout():

    # Clear login session
    session.clear()

    return redirect(url_for("login"))


# ---------------------------------------
# RUN APPLICATION
# ---------------------------------------

if __name__ == "__main__":

    print("--------------------------------------")
    print(" Fake Job Detection Using ML")
    print("--------------------------------------")
    print("Server starting...")
    print("Open http://127.0.0.1:5000")

    app.run(
        debug=True,
        port=5000
    )
