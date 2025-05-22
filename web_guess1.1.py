import streamlit as st
import pickle
import numpy as np
import re
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Bio import SeqIO

@st.cache_resource
def load_model_cached():
    return load_model('model_codons_balanced_50epoch.h5')

@st.cache_data
def load_tokenizer():
    with open('tokenizer_balanced.pkl', 'rb') as handle:
        return pickle.load(handle)

@st.cache_data
def load_label_map():
    with open('label_map_balanced.pkl', 'rb') as handle:
        label_map = pickle.load(handle)
    return {v: k for k, v in label_map.items()}  # Odwrócenie mapy etykiet

model = load_model_cached()
tokenizer = load_tokenizer()
reverse_label_map = load_label_map()

def split_into_codons(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    return " ".join(codons)

def preprocess_sequence(seq, tokenizer, max_length=500):
    codon_seq = split_into_codons(seq)
    encoded_seq = tokenizer.texts_to_sequences([codon_seq])
    padded_seq = pad_sequences(encoded_seq, maxlen=max_length, padding='post')
    return padded_seq

def validate_sequence(seq):
    return re.fullmatch(r'[ATGC]+', seq.upper()) is not None

def process_and_predict(seq):
    processed_sequence = preprocess_sequence(seq, tokenizer)
    prediction = model.predict(processed_sequence)
    predicted_label_index = np.argmax(prediction)
    predicted_bacteria = reverse_label_map[predicted_label_index]
    predicted_probability = prediction[0][predicted_label_index]
    return predicted_bacteria, predicted_probability

st.set_page_config(page_title="RNA Bacteria Classification", layout="centered")

language = st.selectbox("Wybierz język / Select language", ["Polski", "English"])

texts = {
    "Polski": {
        "title": "Klasyfikacja bakterii na podstawie sekwencji DNA",
        "description": "Wprowadź sekwencję kodującą 16S rRNA, aby przewidzieć gatunek bakterii.",
        "input_label": "Wprowadź sekwencję DNA:",
        "analyze_button": "Przeanalizuj sekwencję",
        "warning": "Wprowadź sekwencję",
        "result_label": "**Przewidywana bakteria:**",
        "probability_label": "**Prawdopodobieństwo:**",
        "repo_link": "Mój kod źródłowy: [GitHub Repo](https://github.com/KacperWardzala/NN_master_app)",
        "fasta_section": "Wczytaj plik FASTA",
        "fasta_label": "Załaduj plik FASTA z sekwencjami DNA:",
        "fasta_error": "Sekwencja `{}` zawiera nieprawidłowe znaki. Dozwolone: A, T, G, C.",
        "fasta_predicted": "Przewidywane: {} | Prawdopodobieństwo: {:.4f}"
    },
    "English": {
        "title": "Bacteria Classification Based on DNA Sequence",
        "description": "Enter a 16S rRNA coding sequence to predict bacterial species.",
        "input_label": "Enter DNA sequence:",
        "analyze_button": "Analyze sequence",
        "warning": "Please enter a sequence",
        "result_label": "**Predicted bacteria:**",
        "probability_label": "**Probability:**",
        "repo_link": "My source code: [GitHub Repo](https://github.com/KacperWardzala/NN_master_app)",
        "fasta_section": "FASTA upload",
        "fasta_label": "Upload FASTA file with DNA sequences:",
        "fasta_error": "Sequence `{}` contains invalid characters. Allowed: A, T, G, C only.",
        "fasta_predicted": "Predicted: {} | Probability: {:.4f}"
    }
}

st.title(texts[language]["title"])
st.write(texts[language]["description"])

user_sequence = st.text_input(texts[language]["input_label"], "")

if st.button(texts[language]["analyze_button"]):
    if not user_sequence.strip():
        st.warning(texts[language]["warning"])
    elif not validate_sequence(user_sequence.strip().upper()):
        st.error(texts[language]["fasta_error"].format("Manual sequence"))
    else:
        processed_sequence = preprocess_sequence(user_sequence.strip().upper(), tokenizer)
        prediction = model.predict(processed_sequence)
        predicted_label_index = np.argmax(prediction)
        predicted_bacteria = reverse_label_map[predicted_label_index]
        predicted_probability = prediction[0][predicted_label_index]

        st.success(f"{texts[language]['result_label']} {predicted_bacteria}")
        st.write(f"{texts[language]['probability_label']} {predicted_probability:.4f}")

st.subheader(texts[language]["fasta_section"])
fasta_file = st.file_uploader(texts[language]["fasta_label"], type=["fasta", "fa", "txt"])

if fasta_file:
    sequences = list(SeqIO.parse(io.TextIOWrapper(fasta_file, encoding="utf-8"), "fasta"))
    for record in sequences:
        seq_str = str(record.seq).upper()
        st.markdown(f"**ID:** `{record.id}`")
        
        if validate_sequence(seq_str):
            bacteria, prob = process_and_predict(seq_str)
            st.success(texts[language]["fasta_predicted"].format(bacteria, prob))
        else:
            st.error(texts[language]["fasta_error"].format(record.id))

st.markdown(texts[language]["repo_link"])
st.image("bacteria-6908969_1280.png", caption="Bacteria illustration", use_container_width=True)
