import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model("author_recognition_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

authors = ["Agatha Christie", "Arthur Conan Doyle", "Charles Dickens", "Fyodor Dostoyevski", "Jane Austen"]

def predict_author(text, max_length=500):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)
    predicted_author = authors[np.argmax(prediction)]
    return predicted_author

if __name__ == "__main__":
    while True:
        text_input = input("Podaj fragment tekstu (lub wpisz 'exit', aby zakończyć): ")
        if text_input.lower() == "exit":
            break
        print(f"Przewidywany autor: {predict_author(text_input)}")
