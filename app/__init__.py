# app/__init__.py

import os
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask

def create_app():
    """
    Application factory function. This is where the app is configured
    and all necessary models and resources are loaded and initialized.
    """
    # Create the Flask app instance
    # instance_relative_config=True tells the app that config files are
    # relative to the instance folder.
    app = Flask(__name__, instance_relative_config=True)

    # --- MODEL LOADING, API CONFIG, AND WARM-UP ---
    # This entire block runs only ONCE when the application starts.
    print("--- Initializing Application and Loading Models ---")

    # This builds a robust path to the project's root directory.
    # It allows the app to find its files regardless of where you run it from.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load environment variables from the keys.env file at the project root
    load_dotenv(dotenv_path=os.path.join(project_root, 'keys.env'))
    API_KEY = os.getenv("API_KEY")
    genai.configure(api_key=API_KEY)

    # We store the loaded models and objects in the app.config dictionary.
    # This is the standard Flask way to make resources globally accessible
    # to the application's context (e.g., in the route handlers).
    data_path = os.path.join(project_root, 'data', 'data.json')
    words_path = os.path.join(project_root, 'models', 'words.pkl')
    classes_path = os.path.join(project_root, 'models', 'classes.pkl')
    model_path = os.path.join(project_root, 'models', 'chatbot_model.h5')

    app.config['LEMMATIZER'] = WordNetLemmatizer()
    app.config['INTENTS'] = json.loads(open(data_path).read())
    app.config['WORDS'] = pickle.load(open(words_path, 'rb'))
    app.config['CLASSES'] = pickle.load(open(classes_path, 'rb'))
    app.config['KERAS_MODEL'] = keras.models.load_model(model_path)
    app.config['GEMINI_MODEL'] = genai.GenerativeModel("gemini-2.0-flash")

    # --- WARM-UP SECTION ---
    # This forces the models to initialize fully at startup, not on the first request.
    print("Warming up Keras model...")
    dummy_input = np.zeros((1, len(app.config['WORDS'])))
    app.config['KERAS_MODEL'].predict(dummy_input)
    print("Keras model is warm.")

    print("Warming up lemmatizer...")
    app.config['LEMMATIZER'].lemmatize("test")
    print("Lemmatizer is warm.")
    print("--- Application is ready to serve requests ---")


    # Using a 'with' statement makes the application context available.
    with app.app_context():
        # Import and register the routes from routes.py using a Blueprint
        from .routes import main_bp
        app.register_blueprint(main_bp)

    return app