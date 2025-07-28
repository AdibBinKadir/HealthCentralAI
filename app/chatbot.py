# app/chatbot.py

import random
import numpy as np
import nltk
from flask import current_app

def clean_up_sentence(sentence):
    lemmatizer = current_app.config['LEMMATIZER']
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    words = current_app.config['WORDS']
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s_word in sentence_words:
        for i, word in enumerate(words):
            if word == s_word:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    interpreter = current_app.config['TFLITE_INTERPRETER']
    input_details = current_app.config['TFLITE_INPUT_DETAILS']
    output_details = current_app.config['TFLITE_OUTPUT_DETAILS']
    classes = current_app.config['CLASSES']

    bow = bag_of_words(sentence)
    input_data = np.array([bow], dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list):
    intents_json = current_app.config['INTENTS']
    if not intents_list:
        return "I'm sorry, I don't quite understand. Could you please rephrase?"

    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that. Can you try asking differently?"

def call_gemini_api(prompt):
    gemini_model = current_app.config['GEMINI_MODEL']
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "I seem to be having trouble accessing my advanced knowledge base. Please try again later."

def chatbot_response_smart(user_input, confidence_threshold=0.75):
    predicted_intents = predict_class(user_input)

    if predicted_intents and float(predicted_intents[0]['probability']) >= confidence_threshold:
        return get_response(predicted_intents)

    prompt = (
        "You are a helpful and reassuring first-aid assistant. "
        f"A user said: '{user_input}'. Provide clear, simple, and safe "
        "first-aid advice. If it is not a medical question, provide a helpful "
        "general response. Keep your answer concise."
    )
    return call_gemini_api(prompt)