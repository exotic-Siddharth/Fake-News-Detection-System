# 📰 Fake News Detection using Machine Learning

## 📌 Overview

This project is an NLP-based Fake News Detection System that classifies news articles as **Real** or **Fake** using Machine Learning techniques.

The system uses **TF-IDF Vectorization** and **Logistic Regression**, achieving approximately **98% accuracy** on a dataset containing more than **45,000 news articles**.

---

## 🚀 Features

✅ Fake/Real News Classification  
✅ 98% Accuracy  
✅ Interactive Streamlit Web Application  
✅ Confidence Score Prediction  
✅ NLP Text Preprocessing  
✅ Model Saving and Loading

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- NLP (TF-IDF)
- Logistic Regression

---

## 📂 Dataset

Dataset Used:

- Fake News Dataset (45k+ Articles)

Classes:

- Fake News → 0
- Real News → 1

---

## 📊 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 98% |
| Precision | 98% |
| Recall | 98% |
| F1 Score | 98% |

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── app.py
├── train.py
├── predict.py
├── EDA.ipynb
├── Fake.csv
├── True.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
```

Move into the folder:

```bash
cd fake-news-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

```bash
python train.py
```

---

## ▶️ Run Streamlit App

```bash
python -m streamlit run app.py
```

---

## 🖥 Application Preview

(Add screenshots here)

```md
![Home Page](images/home.png)
```

---

## 📈 Future Improvements

- DistilBERT Implementation
- RoBERTa Implementation
- Model Comparison (SVM, XGBoost, Random Forest)
- Deploy on Streamlit Cloud

---

## 👨‍💻 Author

**Siddharth Patel**

- B.Tech CSE, NIT Patna
- 300+ DSA Problems Solved
- Interested in Machine Learning, Deep Learning and Competitive Programming.