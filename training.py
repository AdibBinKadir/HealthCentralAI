# training.py

import os
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras

# --- Constants ---
# Define paths based on the new project structure
DATA_DIR = "data"
MODELS_DIR = "models"
INTENTS_FILE = os.path.join(DATA_DIR, "data.json")
WORDS_FILE = os.path.join(MODELS_DIR, "words.pkl")
CLASSES_FILE = os.path.join(MODELS_DIR, "classes.pkl")
MODEL_FILE = os.path.join(MODELS_DIR, "chatbot_model.h5")

# --- Model Hyperparameters ---
TRAINING_EPOCHS = 300
BATCH_SIZE = 5
LEARNING_RATE = 0.01


def setup_nltk():
    """
    Downloads all necessary NLTK data packages.
    The NLTK downloader is smart and will not re-download packages
    if they are already present.
    """
    print("Setting up NLTK... (This may take a moment on first run)")
    # This is the corrected, simpler, and more robust way to do it.
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    print("NLTK setup complete.")


def load_and_preprocess_data(intents_path):
    """
    Loads intents data, tokenizes patterns, lemmatizes words,
    and prepares training data.

    Args:
        intents_path (str): The path to the intents JSON file.

    Returns:
        tuple: A tuple containing the processed words, classes, and training data.
    """
    print(f"Loading and preprocessing data from {intents_path}...")
    lemmatizer = WordNetLemmatizer()
    with open(intents_path) as file:
        intents = json.load(file)

    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ".", ","]

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            # Tokenize each word in the sentence
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add documents to the corpus
            documents.append((word_list, intent["tag"]))
            # Add to our classes list
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    # Lemmatize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    print(f"{len(documents)} documents processed.")
    print(f"{len(classes)} classes: {classes}")
    print(f"{len(words)} unique lemmatized words.")

    # --- Create training data ---
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        # Lemmatize pattern words for matching
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
        for word in words:
            bag.append(1) if word in pattern_words else bag.append(0)

        # Create output row with 1 for the current tag
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    
    # Ensure training data is a NumPy array
    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:, 0]), dtype=np.float32)
    train_y = np.array(list(training[:, 1]), dtype=np.float32)
    
    print("Training data created successfully.")
    return words, classes, train_x, train_y


def build_model(input_shape, output_shape):
    """
    Builds, compiles, and returns the Keras Sequential model.

    Args:
        input_shape (int): The shape of the input layer (vocabulary size).
        output_shape (int): The shape of the output layer (number of classes).

    Returns:
        keras.Model: The compiled Keras model.
    """
    print("Building neural network model...")
    model = keras.models.Sequential()
    # Input layer with 128 neurons
    model.add(keras.layers.Dense(128, input_shape=(input_shape,), activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    # Hidden layer with 64 neurons
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.5))
    # Output layer with a neuron for each class, using softmax for probabilities
    model.add(keras.layers.Dense(output_shape, activation="softmax"))

    # Use the SGD optimizer. The `decay` argument is deprecated.
    sgd = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True)

    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    print("Model built and compiled.")
    model.summary()
    return model


def save_artifacts(model, words, classes):
    """Saves the trained model and the processed word/class lists."""
    print("Saving artifacts...")
    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model.save(MODEL_FILE)
    with open(WORDS_FILE, "wb") as f:
        pickle.dump(words, f)
    with open(CLASSES_FILE, "wb") as f:
        pickle.dump(classes, f)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Words saved to {WORDS_FILE}")
    print(f"Classes saved to {CLASSES_FILE}")


def main():
    """Main function to orchestrate the training process."""
    setup_nltk()
    words, classes, train_x, train_y = load_and_preprocess_data(INTENTS_FILE)
    model = build_model(input_shape=len(train_x[0]), output_shape=len(train_y[0]))

    print("--- Starting Model Training ---")
    model.fit(train_x, train_y, epochs=TRAINING_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    print("--- Model Training Complete ---")

    save_artifacts(model, words, classes)


if __name__ == "__main__":
    main()