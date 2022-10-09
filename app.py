import os
import logging
import json

from flask import Flask, request, jsonify, render_template

from model.model import TranslationClassifier

app = Flask(__name__)

# define model path
model_path = './model/final_model.h5'

# create instance
model = TranslationClassifier(model_path)
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/v1/predict", methods=["POST"])
def predict():
    toTranslate = request.form['toTranslate']
    json_str = jsonify({'translated': str(model.testPrint(toTranslate))})
    return (json_str)


def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
