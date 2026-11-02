# Fake News Detection

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Classifies a news article as **real** or **fake** with a TF-IDF + Logistic Regression model, served
through a small Streamlit app that also shows the model's confidence.

## Results

Held-out 20% split of 44,898 articles (21,417 real, 23,481 fake):

| Metric | Score |
|---|---|
| Accuracy | 0.98 |
| Precision | 0.98 |
| Recall | 0.98 |
| F1 | 0.98 |

## How it works

1. `train.py` joins each article's title and body, shuffles the two classes together and makes an 80/20 split.
2. A `TfidfVectorizer` (10,000 features) turns text into sparse vectors.
3. `LogisticRegression` is trained on them, and the classification report is printed.
4. The fitted model and vectorizer are saved as `model.pkl` and `vectorizer.pkl`.
5. `app.py` (Streamlit) and `predict.py` (command line) load both files and classify new text.

## Quick start

```bash
git clone https://github.com/exotic-Siddharth/Fake-News-Detection-System.git
cd Fake-News-Detection-System
pip install -r requirements.txt

python train.py                    # retrain (optional, the saved model is included)
python -m streamlit run app.py     # web app
python predict.py                  # command-line prediction
```

## Project structure

```text
├── app.py             Streamlit web app
├── predict.py         command-line prediction
├── train.py           training and evaluation
├── Fake.csv           fake articles
├── True.csv           real articles
├── model.pkl          trained Logistic Regression
├── vectorizer.pkl     fitted TF-IDF vectorizer
└── requirements.txt
```

## Limitations

- **Source shortcut.** 98.9% of the real articles contain a `(Reuters)` dateline near the start, against
  0.01% of the fake ones, so part of the score comes from recognising the source rather than judging
  the content. Expect lower accuracy on articles from other outlets.
- **Random split.** Near-duplicate stories can land in both train and test, which inflates held-out scores.

## Next steps

- Strip datelines and outlet names before training, and report accuracy with and without them
- Compare against linear SVM, gradient boosting and a fine-tuned DistilBERT
- Evaluate on articles from outside this dataset
- Deploy the app on Streamlit Community Cloud

## Author

**Siddharth Patel**, B.Tech CSE, NIT Patna
