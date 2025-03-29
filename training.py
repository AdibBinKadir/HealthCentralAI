import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.optimizers import SGD

# Download required modules
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data from json file
intents = json.loads(open('data.json').read())

# Initialize lists to store words, classes, and documents
words = []
classes = []
documents = []

# List of characters to ignore in the patterns
ignore_letters = ['?', '!', '.', ',']

# Extract patterns and tags from intents data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize the pattern and store in words list
        word_list = nltk.word_tokenize(pattern)
        words.append(word_list)

        # Append the tokenized pattern and its tag to the documents list
        documents.append((word_list, intent['tag']))

        # Append the tag to classes list, if it's not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word) for sublist in words for word in sublist if word not in ignore_letters]
words = list(set(words)) # set() will remove the duplicates automatically
words.sort()
classes = list(set(classes))
classes.sort()

# Pickle the words and classes for future use
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Prepare bag of words representation for each document
# Prepare bag of words representation for each document
for document in documents:
    word_patterns = set(map(lambda x: lemmatizer.lemmatize(x.lower()), document[0]))
    bag = [1 if word in word_patterns else 0 for word in words]

    output_row = [0] * len(classes)  # Ensure output_row has the correct length
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Convert training data to numpy arrays
train_x = np.array([np.array(item[0]) for item in training], dtype=np.float32)
train_y = np.array([np.array(item[1]) for item in training], dtype=np.float32)

# Build the model
model = keras.models.Sequential()
model.add(keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Initialize the optimizer
sgd = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile the model with the optimizer, loss function, and evaluation metric
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model to the training data with specified parameters
model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size = 5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')