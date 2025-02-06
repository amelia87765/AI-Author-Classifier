import os
import numpy as np
import tensorflow as tf
import nltk
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import l2

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

VOCAB_SIZE = 5000
MAX_LENGTH = 500
EPOCHS = 30
BATCH_SIZE = 32

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return lemmatized_tokens

def load_data(data_folder):
    texts, labels, authors = [], [], sorted(os.listdir(data_folder))
    for label, author in enumerate(authors):
        author_path = os.path.join(data_folder, author)
        if os.path.isdir(author_path):
            for book_folder in os.listdir(author_path):
                book_path = os.path.join(author_path, book_folder)
                if os.path.isdir(book_path):
                    for chapter_file in os.listdir(book_path):
                        if chapter_file.endswith(".txt"):
                            with open(os.path.join(book_path, chapter_file), "r", encoding="utf-8") as file:
                                texts.append(file.read())
                                labels.append(label)
    return texts, labels, authors

def prepare_data(texts, labels, vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH, test_size=0.2):
    preprocessed_texts = [' '.join(preprocess_text(text)) for text in texts]
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(preprocessed_texts)
    sequences = tokenizer.texts_to_sequences(preprocessed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=test_size, random_state=42, stratify=labels
    )
    return X_train, X_test, np.array(y_train), np.array(y_test), tokenizer

def build_lstm_model(vocab_size, num_classes, embedding_dim=128, input_length=MAX_LENGTH):
    model = tf.keras.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        Bidirectional(LSTM(256, return_sequences=False, kernel_regularizer=l2(0.0001))),
        Dense(128, activation="relu", kernel_regularizer=l2(0.0001)),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def train_model(model, X_train, y_train, X_test, y_test, checkpoint_path="author_recognition_model.keras"):
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[model_checkpoint, early_stopping]
    )

    return model, history

def save_plot(fig, filename, folder="plots"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    plt.show()

def plot_learning_curve(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history['loss'], label='Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_title('Krzywa strat')
    ax[0].legend()
    
    ax[1].plot(history['accuracy'], label='Accuracy')
    ax[1].plot(history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title('Krzywa dokładności')
    ax[1].legend()

    save_plot(fig, 'learning_curve.png')

def plot_confusion_matrix(y_true, y_pred, authors):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=authors, yticklabels=authors, ax=ax)
    ax.set_title('Macierz pomyłek')

    save_plot(fig, 'confusion_matrix.png')

def print_author_word_summary(texts, labels, authors, top_n=10):
    author_texts = {author: [] for author in authors}
    for text, label in zip(texts, labels):
        author_texts[authors[label]].append(text)

    for author, texts in author_texts.items():
        all_words = []
        for text in texts:
            tokens = preprocess_text(text)
            all_words.extend(tokens)

        word_counts = Counter(all_words)
        print(f"Najczęściej występujące słowa dla {author}:")
        most_common = word_counts.most_common(top_n)
        for word, count in most_common:
            print(f"{word}: {count}")
        print("-" * 50)

def plot_author_word_frequencies(texts, labels, authors, top_n=10):
    author_texts = {author: [] for author in authors}
    for text, label in zip(texts, labels):
        author_texts[authors[label]].append(text)

    for author, texts in author_texts.items():
        all_words = []
        for text in texts:
            tokens = preprocess_text(text)
            all_words.extend(tokens)

        word_counts = Counter(all_words)
        most_common = word_counts.most_common(top_n)
        
        if not most_common:
            continue

        words, counts = zip(*most_common)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(words, counts, color='skyblue')
        ax.set_title(f'Najczęstsze słowa - {author}')
        ax.set_ylabel('Liczba wystąpień')
        ax.set_xticklabels(words, rotation=45, ha="right")

        save_plot(fig, f'word_freq_{author}.png')
        plt.close(fig)

if __name__ == "__main__":
    data_folder = "Books"
    checkpoint_path = "author_recognition_model.keras"

    texts, labels, authors = load_data(data_folder)
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(texts, labels)

    num_classes = len(authors)
    model = build_lstm_model(VOCAB_SIZE, num_classes=num_classes)

    model, history = train_model(model, X_train, y_train, X_test, y_test, checkpoint_path=checkpoint_path)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model osiągnął dokładność: {accuracy:.2%}")

    model.save("author_recognition_model.keras")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("Model i tokenizer zapisane.")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=authors))

    #print_author_word_summary(texts, labels, authors)
    #plot_author_word_frequencies(texts, labels, authors)
    plot_learning_curve(history.history)
    plot_confusion_matrix(y_test, y_pred, authors)
