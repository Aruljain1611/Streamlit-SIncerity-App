import streamlit as st
import pickle
import tensorflow as tf
import numpy as np

# Load the TFLite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to perform inference
def predict(interpreter, input_data):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]

    return output_data

# Load the tokenizer and model
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model("model_with_custom_ops.tflite")
except FileNotFoundError:
    st.error("Tokenizer or model file not found. Please ensure 'tokenizer.pickle' and 'model_with_custom_ops.tflite' are in the correct directory.")
    st.stop()

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Streamlit UI
st.title("My App: Sincerity Classifier")
st.write("The goal of this project is to develop and deploy a text classification model using TensorFlow. The model will be designed to categorize text data into predefined classes based on its content. Text classification has a wide range of applications, including sentiment analysis, spam detection, topic categorization, and more.")

sentence = st.text_input("Write the sentence you want to check")
threshold = st.slider("Select the threshold for sincerity", min_value=0.0, max_value=1.0, step=0.1)

if st.button("Continue"):
    if sentence:
        sentence_list = [sentence]
        tokenized_sentence = tokenizer.texts_to_sequences(sentence_list)
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sentence, padding="post", maxlen=63)
        st.session_state.prediction = predict(model, padded_sequence)
    else:
        st.write("Please enter a sentence.")

# Display result if a prediction exists
if st.session_state.prediction is not None:
    if st.session_state.prediction > threshold:
        st.write("## The sentence was insincere")
    else:
        st.write("## The sentence was sincere")