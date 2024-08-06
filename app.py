from flask import Flask, request, jsonify
from core.model import predict_fertilizer

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    prediction = predict_fertilizer(data)
    return jsonify({"predictions": prediction})


if __name__ == "__main__":
    app.run(debug=True)
