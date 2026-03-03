# 🌸 Bloom AI | Sentiment Analysis & Scoring Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

**Bloom AI** is an end-to-end sentiment analysis application powered by **Deep Learning (LSTM)**. Unlike traditional classification models that simply label text as "Positive" or "Negative," Bloom AI uses a regression-based approach to predict precise star ratings (1.0 - 5.0) from raw text reviews.

This project demonstrates the full lifecycle of a Machine Learning product: from Data Preprocessing and Model Training to Deployment via an interactive Web UI.

---

## 📸 Interface Preview

### 1. Home & Input Screen
![Home Screen](screenshot/demo0.jpg)

### 2. Analysis Result
![Result Screen](screenshot/demo2.jpg) 

> The application features a custom **"Rose Gold"** UI theme designed for a premium user experience.

### 3. Analytics Dashboard
![Dashboard](screenshot/demo3.jpg)
---

## 🚀 Key Features

* **🧠 LSTM-Based Regression:** Utilizes Long Short-Term Memory networks to understand the context and flow of text, predicting exact sentiment scores (e.g., 4.2/5.0).
* **⚡ Real-Time Inference:** Instant analysis of single user reviews.
* **📂 Batch Processing:** Supports bulk analysis via `.csv` or `.xlsx` file uploads. Process thousands of reviews in seconds.
* **📊 Interactive Dashboard:**
    * Visualizes sentiment distribution.
    * Identifies **"Critical Reviews"** (Score < 2.5) for immediate business action.
* **🎨 Custom Design:** A tailored Streamlit interface with reactive elements and custom CSS styling.

## 🤖 Model Architecture

The model is built to capture semantic meaning and sequence dependencies in text.

1.  **Tokenizer:** Vectorizes text (Top 10,000 words).
2.  **Embedding Layer:** Maps words to dense vectors ($128d$).
3.  **LSTM Layer:** 128 units to capture long-term dependencies in review text.
4.  **Dense Layers:**
    * Hidden Layer: 64 units (ReLU activation).
    * Output Layer: 1 unit (Sigmoid activation).
5.  **Post-Processing:** The model outputs a value between 0-1, which is linearly scaled to the 1-5 star range.

```python
# Model Structure Snippet
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid")) # Regression Output

Made with ❤️ by Aslı
