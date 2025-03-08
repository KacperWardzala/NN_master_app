import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model_cached():
    return load_model('model_codons_lstm128.h5')

@st.cache_data
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

@st.cache_data
def load_label_map():
    with open('label_map.pkl', 'rb') as handle:
        label_map = pickle.load(handle)
    return {v: k for k, v in label_map.items()}  # Odwrócenie mapy etykiet

# Wczytaj model i tokenizator
model = load_model_cached()
tokenizer = load_tokenizer()
reverse_label_map = load_label_map()

# Funkcja do dzielenia sekwencji na kodony (trójki nukleotydów)
def split_into_codons(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    return " ".join(codons)

# Funkcja do przetwarzania sekwencji RNA przed predykcją
def preprocess_sequence(seq, tokenizer, max_length=500):
    codon_seq = split_into_codons(seq)
    encoded_seq = tokenizer.texts_to_sequences([codon_seq])
    padded_seq = pad_sequences(encoded_seq, maxlen=max_length, padding='post')
    return padded_seq

# Interfejs użytkownika w Streamlit
st.title("Klasyfikacja bakterii na podstawie sekwencji RNA")
st.write("Wprowadź sekwencję 16S rRNA, aby przewidzieć przynależność bakteryjną.")

# 🟢 Nowa wersja: `st.text_input` + `st.button`
user_sequence = st.text_input("Wprowadź sekwencję RNA:", "")

if st.button("Przeanalizuj sekwencję"):
    if not user_sequence.strip():
        st.warning("Wprowadź sekwencję")
    else:
        processed_sequence = preprocess_sequence(user_sequence, tokenizer)
        prediction = model.predict(processed_sequence)
        predicted_label_index = np.argmax(prediction)
        predicted_bacteria = reverse_label_map[predicted_label_index]
        predicted_probability = prediction[0][predicted_label_index]

        st.success(f"**Przewidywana bakteria:** {predicted_bacteria}")
        st.write(f"**Prawdopodobieństwo:** {predicted_probability:.4f}")
