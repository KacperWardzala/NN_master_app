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
st.set_page_config(page_title="RNA Bacteria Classification", layout="centered")

# Wybór języka
language = st.selectbox("Wybierz język / Select language", ["Polski", "English"])

# Teksty dla obu języków
texts = {
    "Polski": {
        "title": "Klasyfikacja bakterii na podstawie sekwencji RNA",
        "description": "Wprowadź sekwencję 16S rRNA, aby przewidzieć przynależność bakteryjną.",
        "input_label": "Wprowadź sekwencję RNA:",
        "analyze_button": "Przeanalizuj sekwencję",
        "warning": "Wprowadź sekwencję",
        "result_label": "**Przewidywana bakteria:**",
        "probability_label": "**Prawdopodobieństwo:**",
        "repo_link": "Mój kod źródłowy: [GitHub Repo](https://github.com/KacperWardzala/NN_master_app)"
    },
    "English": {
        "title": "Bacteria Classification Based on RNA Sequence",
        "description": "Enter a 16S rRNA sequence to predict bacterial affiliation.",
        "input_label": "Enter RNA sequence:",
        "analyze_button": "Analyze sequence",
        "warning": "Please enter a sequence",
        "result_label": "**Predicted bacteria:**",
        "probability_label": "**Probability:**",
        "repo_link": "My source code: [GitHub Repo](https://github.com/KacperWardzala/NN_master_app)"
    }
}

st.title(texts[language]["title"])
st.write(texts[language]["description"])

user_sequence = st.text_input(texts[language]["input_label"], "")

if st.button(texts[language]["analyze_button"]):
    if not user_sequence.strip():
        st.warning(texts[language]["warning"])
    else:
        processed_sequence = preprocess_sequence(user_sequence, tokenizer)
        prediction = model.predict(processed_sequence)
        predicted_label_index = np.argmax(prediction)
        predicted_bacteria = reverse_label_map[predicted_label_index]
        predicted_probability = prediction[0][predicted_label_index]

        st.success(f"{texts[language]['result_label']} {predicted_bacteria}")
        st.write(f"{texts[language]['probability_label']} {predicted_probability:.4f}")

# Link do repozytorium
st.markdown(texts[language]["repo_link"])
