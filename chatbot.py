import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import google.generativeai as genai

API_KEY = "AIzaSyAs8o2FP3RjZBPVoZrUJ67svphuZ3w_xOY"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Load necessary resources
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
keras_model = keras.models.load_model('chatbot_model.h5') # Renamed to avoid confusion with gemini_model

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to create bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class (modified to always return top intent)
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = keras_model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res)] # Keep all results
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    if results:
        return_list.append({'intent': classes[results[0][0]], 'probability': results[0][1]})
    return return_list

# Function to get a response from the intents data
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return None # Return None if no matching intent found (shouldn't happen with predict_class modification)

# --- Integration with Gemini API ---
def call_gemini_api(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# --- Smart Response Function ---
def chatbot_response_smart(user_input, confidence_threshold=0.7):
    ints = predict_class(user_input)
    if ints:
        top_intent = ints[0]
        if top_intent['probability'] >= confidence_threshold:
            # Keras model is confident, use its response
            keras_response = get_response(ints, intents)
            if keras_response:
                return keras_response
            else:
                # Fallback to Gemini if no response found in intents (shouldn't happen usually)
                prompt = f"Provide first aid advice for: {user_input}"
                return call_gemini_api(prompt)
        else:
            # Keras model is not confident, use Gemini
            prompt = f"Provide first aid advice for: {user_input}"
            return call_gemini_api(prompt)
    else:
        # No intent predicted, use Gemini as a fallback
        prompt = f"Provide first aid advice for: {user_input}"
        return call_gemini_api(prompt)

# --- Chatbot Interaction Loop ---
print("Chatbot is running in smart mode!")
# while True:
#     message = input("> ")
#     if message.lower() == "quit":
#         break

#     response = chatbot_response_smart(message)
#     print(response)