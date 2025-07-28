# app/chatbot.py

import random
import numpy as np
import nltk
from flask import current_app


def clean_up_sentence(sentence):
    """
    Tokenizes and lemmatizes a sentence.

    Args:
        sentence (str): The input sentence from the user.

    Returns:
        list: A list of lemmatized words.
    """
    lemmatizer = current_app.config['LEMMATIZER']
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    """
    Converts a sentence into a bag-of-words numpy array.

    Args:
        sentence (str): The user's input sentence.

    Returns:
        np.array: A numpy array representing the sentence as a bag of words.
    """
    # Get the pre-loaded vocabulary from the app config
    words = current_app.config['WORDS']
    sentence_words = clean_up_sentence(sentence)
    # Create a bag of zeros with the same length as the vocabulary
    bag = [0] * len(words)
    for s_word in sentence_words:
        for i, word in enumerate(words):
            if word == s_word:
                # Set 1 if the current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    """
    Predicts the intent class for a given sentence.

    Args:
        sentence (str): The user's input sentence.

    Returns:
        list: A list containing a dictionary of the predicted intent and its probability.
              Returns an empty list if no prediction is made.
    """
    # Get pre-loaded models from the app config
    keras_model = current_app.config['KERAS_MODEL']
    classes = current_app.config['CLASSES']

    # Generate the bag of words for the input sentence
    bow = bag_of_words(sentence)
    # Predict using the Keras model
    res = keras_model.predict(np.array([bow]))[0]

    # Create a list of intents and their probabilities
    results = [[i, r] for i, r in enumerate(res)]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    if results:
        # Append the top result to the return list
        return_list.append(
            {"intent": classes[results[0][0]], "probability": str(results[0][1])}
        )
    return return_list


def get_response(intents_list):
    """
    Selects a random response from the intents file based on the predicted tag.

    Args:
        intents_list (list): The list returned by the predict_class function.

    Returns:
        str: A random response string from the matched intent, or None if no match.
    """
    # Get the pre-loaded intents JSON from the app config
    intents_json = current_app.config['INTENTS']

    if not intents_list:
        return "I'm sorry, I don't understand. Could you rephrase?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            return result
    return None


def call_gemini_api(prompt):
    """
    Calls the Gemini API with a specific prompt and returns the response.

    Args:
        prompt (str): The prompt to send to the Gemini model.

    Returns:
        str: The text response from the API or an error message.
    """
    # Get the pre-loaded Gemini model from the app config
    gemini_model = current_app.config['GEMINI_MODEL']
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        # In a real app, you might want to log this error more formally
        print(f"Gemini API Error: {e}")
        return "I am having trouble connecting to my advanced knowledge base right now. Please try again."


def chatbot_response_smart(user_input, confidence_threshold=0.75):
    """
    Determines the best response, either from the local model or the Gemini API.

    This is the main function called by the web routes. It orchestrates the process:
    1. Predicts the intent of the user's message.
    2. If the confidence is high, it uses a predefined response.
    3. If confidence is low or the intent is unknown, it falls back to the Gemini API
       to generate a more dynamic response.

    Args:
        user_input (str): The raw text input from the user.
        confidence_threshold (float): The probability threshold to trust the local model.

    Returns:
        str: The final chatbot response.
    """
    # Predict the intent using the local Keras model
    predicted_intents = predict_class(user_input)

    if predicted_intents:
        top_intent = predicted_intents[0]
        # Check if the model's confidence is above the threshold
        if float(top_intent['probability']) >= confidence_threshold:
            # If confident, get a predefined response
            return get_response(predicted_intents)

    # If the model is not confident, or no intent was predicted,
    # formulate a prompt and call the Gemini API as a fallback.
    prompt = f"You are a helpful and reassuring first-aid assistant. A user said: '{user_input}'. Provide clear, simple, and safe first-aid advice for this. If it sounds like a non-medical question, provide a helpful general response."
    return call_gemini_api(prompt)