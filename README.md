# 🌍 Maghreb Dialect Detection & Sentiment Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-Arabic-green?style=for-the-badge"/>
</p>

<p align="center">
  An end-to-end NLP application that automatically detects whether a text is written in <strong>Moroccan (Darija)</strong>, <strong>Algerian</strong>, or <strong>Tunisian</strong> dialect — and then performs <strong>sentiment analysis</strong> on it.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Models & Approach](#-models--approach)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Demo](#-demo)
- [Author](#-author)

---

## 🧠 Overview

Arabic NLP is a challenging field due to the diversity of dialects across the Arab world. The Maghreb region (Morocco, Algeria, Tunisia) has particularly underrepresented dialects in existing NLP research. This project addresses that gap by building a **dialect identification + sentiment analysis pipeline** specifically for Maghrebi Arabic text.

---

## ❓ Problem Statement

- Existing Arabic NLP tools are mostly trained on **Modern Standard Arabic (MSA)** and fail on dialectal text.
- Moroccan Darija, Algerian Arabic, and Tunisian Arabic differ significantly in vocabulary, morphology, and code-switching (mixing with French/Berber).
- There is a real need for tools that can process these dialects for **social media monitoring, customer feedback analysis, and business intelligence**.

---

## ✨ Features

- 🔍 **Dialect Detection** — Identifies whether the input text is Moroccan, Algerian, or Tunisian
- 💬 **Sentiment Analysis** — Classifies sentiment as Positive, Negative, or Neutral
- 🤖 **Multiple Models** — Compares Classical ML, LSTM, and BERT-based approaches
- 🖥️ **Interactive UI** — User-friendly Streamlit web application
- 📊 **Model Comparison Dashboard** — Visualizes performance metrics across models

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| NLP | HuggingFace Transformers, NLTK, spaCy |
| Deep Learning | PyTorch, Keras |
| Classical ML | Scikit-learn (SVM, Naive Bayes) |
| Models | AraBERT / DarijaBERT, LSTM/RNN |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |

---

## 🤖 Models & Approach

The project experiments with **three levels of NLP complexity**:

### 1. 🟢 Classical Machine Learning
- **Models:** Naive Bayes, Support Vector Machine (SVM)
- **Features:** TF-IDF vectorization on preprocessed Arabic text
- **Preprocessing:** Normalization, stop word removal, tokenization

### 2. 🟡 Deep Learning — LSTM / RNN
- **Architecture:** Bidirectional LSTM with embedding layer
- **Training:** Custom-trained on Maghrebi dialect datasets
- **Input:** Padded tokenized sequences

### 3. 🔴 Transformer — AraBERT / DarijaBERT
- **Model:** Pre-trained Arabic BERT fine-tuned on dialectal data
- **Approach:** Transfer learning with fine-tuning on task-specific data
- **Tokenizer:** AraBERT tokenizer with Arabic-aware subword units

---

## 📊 Results

### Dialect Detection

| Model | Accuracy | F1-Score (Macro) |
|---|---|---|
| Naive Bayes | — % | — |
| SVM | — % | — |
| LSTM | — % | — |
| AraBERT / DarijaBERT | — % | — |

### Sentiment Analysis

| Model | Accuracy | F1-Score (Macro) |
|---|---|---|
| Naive Bayes | — % | — |
| SVM | — % | — |
| LSTM | — % | — |
| AraBERT / DarijaBERT | — % | — |

> 📝 **Note:** Fill in your actual scores from your experiments above.

---

## 📁 Project Structure

```
maghreb-dialect-detection/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit application
│
├── data/
│   ├── raw/                      # Raw dialect datasets
│   └── processed/                # Cleaned & preprocessed data
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_classical_ml.ipynb     # SVM & Naive Bayes experiments
│   ├── 03_lstm_model.ipynb       # LSTM/RNN training
│   └── 04_bert_finetuning.ipynb  # AraBERT fine-tuning
│
├── models/
│   └── saved/                    # Saved model weights & checkpoints
│
├── src/
│   ├── preprocessing.py          # Text cleaning & normalization
│   ├── features.py               # Feature extraction (TF-IDF, embeddings)
│   └── evaluate.py               # Evaluation metrics & plots
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/MayssaeATIFI/maghreb-dialect-detection.git
cd maghreb-dialect-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

Then open your browser at `http://localhost:8501`

---

## 🎬 Demo

> 📸 *Add a screenshot or GIF of your Streamlit app here*

```
[Screenshot of the app interface]
```

**Example input:**
```
واش كاين شي حاجة زوينة في هاد المنتوج؟
```
**Output:**
- 🌍 Detected Dialect: **Moroccan (Darija)**
- 💬 Sentiment: **Positive**

---

## 👩‍💻 Author

**Mayssae ATIFI**
ML Engineer | Data Scientist | NLP Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/mayssae-atifi)
[![Email](https://img.shields.io/badge/Email-mayssaeatifi86@gmail.com-red?style=flat&logo=gmail)](mailto:mayssaeatifi86@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and share.

---

<p align="center">⭐ If you found this project useful, please give it a star!</p>
