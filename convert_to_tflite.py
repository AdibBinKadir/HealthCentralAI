import tensorflow as tf
import numpy as np

print("Loading the original Keras model from 'chatbot_model.h5'...")
# This loads your existing, fully-trained model
model = tf.keras.models.load_model('models/chatbot_model.h5')
print("Model loaded successfully.")

# Initialize the TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# This is the key step: enable default optimizations, which includes
# quantization. This shrinks the model and makes it faster.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting the model to TensorFlow Lite format...")
# Perform the conversion
tflite_quant_model = converter.convert()
print("Conversion complete.")

# Save the new, optimized model file
with open('models/chatbot_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("\nSUCCESS: 'chatbot_model.tflite' has been created.")
print("This new file is much smaller and should be used for deployment.")