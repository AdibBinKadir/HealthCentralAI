import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from keys.env
load_dotenv('keys.env')

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in keys.env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Load necessary resources
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('data.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = keras.models.load_model('chatbot_model.h5')

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

# Function to predict the class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a response from the intents data
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    return "I don't understand."

# --- Integration with Gemini API ---
# Replace this with your actual Gemini API interaction logic
def call_gemini_api(prompt):
    
    try:
        # Generate content
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"An error occurred: {e}"

# --- Integration Approach 1: Enhanced Response Generation ---
def chatbot_response_enhanced(user_input):
    ints = predict_class(user_input)
    if ints:
        tag = ints[0]['intent']
        for intent in intents['intents']:
            if intent['tag'] == tag:
                if "complex_advice" in intent.get('flags', []): # Example flag for using Gemini
                    prompt = f"Provide more detailed first aid advice for: {user_input}"
                    return call_gemini_api(prompt)
                else:
                    return get_response(ints, intents)
    else:
        return "I'm not sure what you mean."

# --- Integration Approach 2: Hybrid Approach (Gemini handles specific intents) ---
def chatbot_response_hybrid(user_input):
    ints = predict_class(user_input)
    if ints:
        tag = ints[0]['intent']
        if tag == "complex_situation": # Example intent that you want Gemini to handle
            prompt = f"A user is asking about: {user_input}. Provide comprehensive first aid guidance."
            return call_gemini_api(prompt)
        else:
            return get_response(ints, intents)
    else:
        return "I'm not sure what you mean."

# --- Chatbot Interaction Loop ---
print("Chatbot is running!")
while True:
    message = input("> ")
    if message.lower() == "quit":
        break

    # Choose which response function to use:
    # response = chatbot_response_enhanced(message)
    response = chatbot_response_hybrid(message)

    print(response)