Opis projektu
Projekt wykorzystuje sieci neuronowe LSTM do klasyfikacji autorów tekstów literackich na podstawie ich stylu pisania. Model został wytrenowany na zbiorze tekstów różnych autorów i zapisany, co pozwala na jego późniejsze użycie bez konieczności ponownego treningu.

Architektura Modelu:

Wejście: Tekst przekształcony za pomocą Tokenizera
Embedding Layer: Reprezentacja słów w postaci wektorów
Bidirectional LSTM Layer: Analiza sekwencji tekstu w obu kierunkach, wychwytująca kontekst słów
Dense Layer (ReLU): Ekstrakcja istotnych cech stylu pisania
Output Layer (Softmax): Klasyfikacja do konkretnego autora

Najlepsza dokładność modelu: 94.44%

Struktura repozytorium
train_model.py – kod do trenowania modelu (nie jest wymagany do samego użycia)
predict_author.py – skrypt do przewidywania autora tekstu
author_recognition_model.keras – zapisany model
tokenizer.pkl – tokenizer używany podczas treningu

Uruchomienie:
pip install -r requirements.txt
python predict_author.py
