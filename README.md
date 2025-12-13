# 📌 IMDB Sentiment Analysis using RNN & GRU

This project performs **Sentiment Analysis** on the IMDB movie reviews dataset using **Neural Networks (RNN & GRU)**.  
The goal is to classify each review as **Positive** or **Negative** using deep learning.

---

## ⭐ Project Overview

This project demonstrates how to:

- Clean and preprocess text  
- Tokenize & pad sequences  
- Build RNN / GRU models  
- Train deep learning models on NLP data  
- Evaluate model performance  
- Predict sentiment for new text  

We first try a Simple RNN model. If accuracy is low, we upgrade to a GRU model for better results.

---

## 📂 Dataset

**IMDB Dataset (50,000 reviews)** 

Format:  
- review → text  
- sentiment → positive / negative  

Dataset is cleaned and prepared before feeding it into the neural network.
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
---

## 🔧 Technologies Used

- Python  
- TensorFlow / Keras  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 🧹 Data Preprocessing

Steps applied:

1. Convert text to lowercase  
2. Remove HTML tags  
3. Keep letters, numbers, and apostrophes only  
4. Remove extra spaces  
5. Tokenize text  
6. Pad sequences to fixed length (`max_len = 250`)

Example cleaning function:

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

---

## 🧠 Model Architecture (Final GRU Model)

```
Input Layer (shape=max_len)
Embedding Layer (vocab_size=20000, output_dim=256)
GRU Layer (128 units, dropout=0.2, recurrent_dropout=0.2)
Dense Layer (128 units, activation='relu')
Dropout (0.3)
Dense Layer (64 units, activation='relu')
Dropout (0.3)
Output Layer (Sigmoid)
```

Loss: **Binary Crossentropy**  
Optimizer: **Adam**  
Metric: **Accuracy**

---

## 🚀 Training

- Epochs: 8  
- Batch Size: 64  
- EarlyStopping enabled  
- Validation Split: 20%  

Training logs example:

```
Epoch 1: val_accuracy = 0.83
Epoch 2: val_accuracy = 0.87
Epoch 3: val_accuracy = 0.88
```

---

## 📊 Final Results

The GRU model achieved:

### ✅ **Test Accuracy = 88.24%**

This is strong performance for sentiment analysis using GRU without pretrained embeddings.

---

## 🧪 Testing the Model

Use this function to test custom text:

```python
def predict_review(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)[0][0]
    
    if pred >= 0.5:
        print(f"Positive Review ({pred:.4f})")
    else:
        print(f"Negative Review ({pred:.4f})")
```

Example:

```python
predict_review("This movie was amazing, I loved it!")
predict_review("The movie was boring and terrible.")
```

---

## ▶️ How to Run

1. Upload the IMDB dataset file  
2. Run preprocessing  
3. Tokenize & pad sequences  
4. Train GRU model  
5. Evaluate results  
6. Predict new reviews  

---

## 📝 Conclusion

- Simple RNN gave weak performance  
- GRU improved accuracy significantly  
- Final accuracy reached **88%**  
- No pretrained embeddings used  
- Perfect for NLP projects and university assignments  
