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

bacteria_df = pd.read_csv("balanced_metadata.csv")

def load_fasta_sequences(fasta_file):
    sequences = []
    sequence_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))  
        sequence_ids.append(record.id)   
    return sequences, sequence_ids

def split_into_codons(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    return " ".join(codons)

fasta_file = "balanced_sequences.fasta"

sequences, sequence_ids = load_fasta_sequences(fasta_file)

codon_sequences = [split_into_codons(seq) for seq in sequences]

sequence_to_label = {seq_id: name for seq_id, name in zip(bacteria_df['Sequence_ID'], bacteria_df['Organism_Name'])}
labels = [sequence_to_label[seq_id] for seq_id in sequence_ids]

label_map = {name: idx for idx, name in enumerate(bacteria_df['Organism_Name'].unique())}
numeric_labels = [label_map[label] for label in labels]

with open('label_map_balanced.pkl', 'wb') as handle:
    pickle.dump(label_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

one_hot_labels = to_categorical(numeric_labels, num_classes=len(label_map))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(codon_sequences)
encoded_sequences = tokenizer.texts_to_sequences(codon_sequences)

with open('tokenizer_balanced.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

max_sequence_length = 500
padded_sequences = pad_sequences(encoded_sequences, padding='post', maxlen=max_sequence_length)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.2, random_state=42)

print(f"Kształt danych treningowych: {X_train.shape}")
print(f"Kształt danych testowych: {X_test.shape}")

vocab_size = len(tokenizer.word_index) + 1 
embedding_dim = 128                         
num_classes = len(label_map)                

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(256)),
    Dropout(0.3),
    Dense(128, activation='relu'), 
    Dropout(0.3),
    Dense(num_classes, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,       
    batch_size=128,   
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback, early_stopping]
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

test_log_dir = os.path.join(log_dir, "test")
test_writer = tf.summary.create_file_writer(test_log_dir)

with test_writer.as_default():
    tf.summary.scalar('Test Loss', test_loss, step=0)
    tf.summary.scalar('Test Accuracy', test_acc, step=0)

model.summary()

model.save('model_codons_balanced_50epoch.h5')

with open('training_history_50epoch.pkl', 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
