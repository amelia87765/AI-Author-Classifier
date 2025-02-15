Opis projektu<br>
Projekt wykorzystuje sieci neuronowe LSTM do klasyfikacji autorów tekstów literackich na podstawie ich stylu pisania. Model został wytrenowany na zbiorze tekstów różnych autorów i zapisany, co pozwala na jego późniejsze użycie bez konieczności ponownego treningu.

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
