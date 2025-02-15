Project Description<br>
This project utilizes LSTM neural networks to classify the authors of literary texts based on their writing style. The model has been trained on a dataset of texts from various authors and saved, allowing for later use without the need for retraining. Additionally, the project includes an analysis of the 10 most frequently used words for each author.

Model Architecture:<br>
Input: Text processed using a tokenizer<br>
Embedding Layer: Word representation in vector form<br>
Bidirectional LSTM Layer: Sequential text analysis in both directions, capturing word context<br>
Dense Layer (ReLU): Extraction of essential writing style features<br>
Output Layer (Softmax): Classification into a specific author<br>

Best model accuracy: 94.44%

Repository Structure:<br>
train_model.py – Script for training the model (not required for usage)<br>
predict_author.py – Script for predicting the author of a given text<br>
author_recognition_model.keras – Saved model file<br>
tokenizer.pkl – Tokenizer used during training

Usage:<br>
pip install -r requirements.txt<br>
python predict_author.py  

Opis projektu<br>
Projekt wykorzystuje sieci neuronowe LSTM do klasyfikacji autorów tekstów literackich na podstawie ich stylu pisania. Model został wytrenowany na zbiorze tekstów różnych autorów i zapisany, co pozwala na jego późniejsze użycie bez konieczności ponownego treningu. Projekt dodatkowo uwzględnia zestawienie 10 najczęściej używanych słów dla każdego autora.

Architektura Modelu:<br>
Wejście: Tekst przekształcony za pomocą Tokenizera<br>
Embedding Layer: Reprezentacja słów w postaci wektorów<br>
Bidirectional LSTM Layer: Analiza sekwencji tekstu w obu kierunkach, wychwytująca kontekst słów<br>
Dense Layer (ReLU): Ekstrakcja istotnych cech stylu pisania<br>
Output Layer (Softmax): Klasyfikacja do konkretnego autora<br>

Najlepsza dokładność modelu: 94.44%

Struktura repozytorium<br>
train_model.py – kod do trenowania modelu (nie jest wymagany do samego użycia)<br>
predict_author.py – skrypt do przewidywania autora tekstu<br>
author_recognition_model.keras – zapisany model<br>
tokenizer.pkl – tokenizer używany podczas treningu<br>

Uruchomienie:<br>
pip install -r requirements.txt<br>
python predict_author.py
