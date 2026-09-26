from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")


def predict_job_posting(job_description, job_url=""):

    # Combine job description and URL
    text = f"{job_description} {job_url}"

    # Convert text into TF-IDF
    text_tfidf = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_tfidf)[0]

    # Get probability
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = max(probabilities) * 100

    # Dataset:
    # 0 = Genuine
    # 1 = Fake

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


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Analyze job
@app.route("/api/analyze", methods=["POST"])
def analyze():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Invalid request."
            }), 400

        job_description = data.get("job_description", "")
        job_url = data.get("job_url", "")

        if not job_description and not job_url:
            return jsonify({
                "error": "Please enter a job description."
            }), 400

        result = predict_job_posting(
            job_description,
            job_url
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


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