# app/__init__.py

import os
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    print("--- Initializing Application and Loading Models ---")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(dotenv_path=os.path.join(project_root, 'keys.env'))
    API_KEY = os.getenv("API_KEY")
    genai.configure(api_key=API_KEY)

    data_path = os.path.join(project_root, 'data', 'data.json')
    words_path = os.path.join(project_root, 'models', 'words.pkl')
    classes_path = os.path.join(project_root, 'models', 'classes.pkl')
    model_path = os.path.join(project_root, 'models', 'chatbot_model.tflite')

    app.config['LEMMATIZER'] = WordNetLemmatizer()
    app.config['INTENTS'] = json.loads(open(data_path).read())
    app.config['WORDS'] = pickle.load(open(words_path, 'rb'))
    app.config['CLASSES'] = pickle.load(open(classes_path, 'rb'))
    app.config['GEMINI_MODEL'] = genai.GenerativeModel("gemini-2.0-flash")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    app.config['TFLITE_INTERPRETER'] = interpreter
    app.config['TFLITE_INPUT_DETAILS'] = interpreter.get_input_details()
    app.config['TFLITE_OUTPUT_DETAILS'] = interpreter.get_output_details()

    print("Warming up TFLite model...")
    dummy_input = np.zeros((1, len(app.config['WORDS'])), dtype=np.float32)
    interpreter.set_tensor(app.config['TFLITE_INPUT_DETAILS'][0]['index'], dummy_input)
    interpreter.invoke()
    print("TFLite model is warm.")

    print("Warming up lemmatizer...")
    app.config['LEMMATIZER'].lemmatize("test")
    print("Lemmatizer is warm.")
    print("--- Application is ready to serve requests ---")

    with app.app_context():
        from .routes import main_bp
        app.register_blueprint(main_bp)

    return app