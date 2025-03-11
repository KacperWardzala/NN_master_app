# DNA Bacteria Classification

Streamlit application for classifying bacteria based on DNA sequences. It utilizes an LSTM neural network model trained on DNA sequences of ribosomal rRNA genes of bacteria from the **GeneBank** database. The application is part of the project on which my master's thesis is based.

## 📌 Features
- Predicting the most probable bacterial species/genus based on the entered DNA sequence
- Support for two languages: **Polish** and **English**
- Interactive user interface based on **Streamlit**

## 📂 Project Structure
├── app.py                 # Main Streamlit application file
├── model_codons_lstm128.h5 # Neural network model (private)
├── tokenizer.pkl          # Tokenizer for DNA sequence processing
├── label_map.pkl          # Label map for bacterial classification
├── bacteria-6908969_1280.png # Image used in the application
├── requirements.txt       # List of required libraries
├── README.md              # Project documentation

## 📜 Code Description
The main application file is `web_guess.py`, which contains:

### 1️⃣ Loading Model and Resources
```python
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
    return {v: k for k, v in label_map.items()}  # Reverse label map
```
These functions handle loading the model, tokenizer, and label map, utilizing caching mechanisms to optimize performance.

### 2️⃣ DNA Sequence Processing
```python
def split_into_codons(seq):
    codons = [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
    return " ".join(codons)

def preprocess_sequence(seq, tokenizer, max_length=500):
    codon_seq = split_into_codons(seq)
    encoded_seq = tokenizer.texts_to_sequences([codon_seq])
    padded_seq = pad_sequences(encoded_seq, maxlen=max_length, padding='post')
    return padded_seq
```
Here, the DNA sequence is divided into codons (triplets of nucleotides), tokenized, and normalized for the neural network model.

### 3️⃣ User Interface in Streamlit
```python
st.set_page_config(page_title="DNA Bacteria Classification", layout="centered")

language = st.selectbox("Wybierz język / Select language", ["Polski", "English"])
```
The application allows the user to select a language and enter a DNA sequence for analysis.

```python
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
```
Upon clicking the button, the application processes the data, makes a prediction, and displays the result.

## 📩 Contact
If you have any questions or suggestions, contact me via email: kac.wardzala@gmail.com.

