import os
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Krok 1: Wczytanie pliku CSV z nazwami bakterii
bacteria_df = pd.read_csv("balanced_metadata.csv")

# Krok 2: Wczytanie sekwencji z pliku FASTA
def load_fasta_sequences(fasta_file):
    sequences = []
    sequence_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))  # Dodaj sekwencję jako string
        sequence_ids.append(record.id)       # Zapisz ID sekwencji
    return sequences, sequence_ids

# Funkcja do dzielenia sekwencji na kodony (grupy 3 nukleotydów)
def split_into_codons(seq):
    # Dzielimy sekwencję na pełne kodony
    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    # Łączymy kodony w "zdanie" oddzielone spacjami
    return " ".join(codons)

# Ścieżka do pliku FASTA
fasta_file = "balanced_sequences.fasta"

# Wczytanie sekwencji DNA z pliku FASTA
sequences, sequence_ids = load_fasta_sequences(fasta_file)

# Przekształcenie sekwencji na sekwencje kodonowe
codon_sequences = [split_into_codons(seq) for seq in sequences]

# Krok 3: Przypisanie etykiet (mapowanie nazw bakterii na numeryczne etykiety)
sequence_to_label = {seq_id: name for seq_id, name in zip(bacteria_df['Sequence_ID'], bacteria_df['Organism_Name'])}
labels = [sequence_to_label[seq_id] for seq_id in sequence_ids]

# Mapowanie nazw bakterii na unikalne liczby
label_map = {name: idx for idx, name in enumerate(bacteria_df['Organism_Name'].unique())}
numeric_labels = [label_map[label] for label in labels]

# Zapisanie mapowania etykiet
with open('label_map_balanced.pkl', 'wb') as handle:
    pickle.dump(label_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Krok 4: Jedno gorące kodowanie etykiet
one_hot_labels = to_categorical(numeric_labels, num_classes=len(label_map))

# Krok 5: Tokenizacja sekwencji na poziomie kodonów
# Używamy domyślnego tokenizatora, który dzieli tekst na słowa oddzielone spacjami
tokenizer = Tokenizer()
tokenizer.fit_on_texts(codon_sequences)
encoded_sequences = tokenizer.texts_to_sequences(codon_sequences)

# Zapisanie tokenizatora
with open('tokenizer_balanced.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Padding - teraz maksymalna długość określana jest liczbą kodonów
# Jeśli oryginalnie ustawiono 1500 nukleotydów, to odpowiada to 1500/3 = 500 kodonom
max_sequence_length = 500
padded_sequences = pad_sequences(encoded_sequences, padding='post', maxlen=max_sequence_length)

# Krok 6: Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.2, random_state=42)

# Wyświetlenie kształtu danych
print(f"Kształt danych treningowych: {X_train.shape}")
print(f"Kształt danych testowych: {X_test.shape}")

# Parametry modelu
vocab_size = len(tokenizer.word_index) + 1  # Rozmiar słownika (dodaj 1 dla indeksu "0")
embedding_dim = 128                         # Wymiar przestrzeni osadzania
num_classes = len(label_map)                # Liczba unikalnych klas (bakterii)

# Architektura modelu
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    Bidirectional(LSTM(256, return_sequences=True)),  # Dwukierunkowe LSTM
    Dropout(0.3),
    Bidirectional(LSTM(256)),  # Kolejne LSTM
    Dropout(0.3),
    Dense(128, activation='relu'),  # Warstwa gęsta
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Warstwa wyjściowa z softmax dla wieloklasowej klasyfikacji
])

# Kompilacja modelu
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Dla klasyfikacji wieloklasowej
    metrics=['accuracy']
)

# Krok 7: TensorBoard i EarlyStopping
# Tworzenie logów dla TensorBoard
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Trening modelu
history = model.fit(
    X_train, y_train,
    epochs=50,        # Liczba epok
    batch_size=128,    # Rozmiar batcha
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback, early_stopping]
)

# Ewaluacja modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Testowa strata: {test_loss}, Testowa dokładność: {test_acc}")

# Tworzenie logów dla ewaluacji
test_log_dir = os.path.join(log_dir, "test")
test_writer = tf.summary.create_file_writer(test_log_dir)

# Zapisywanie wyników ewaluacji do TensorBoard
with test_writer.as_default():
    tf.summary.scalar('Test Loss', test_loss, step=0)
    tf.summary.scalar('Test Accuracy', test_acc, step=0)

# Wyświetlenie architektury modelu
model.summary()




# Po zakończeniu trenowania modelu zapisujemy model do pliku .h5
model.save('model_codons_balanced_50epoch.h5')


with open('training_history_50epoch.pkl', 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
