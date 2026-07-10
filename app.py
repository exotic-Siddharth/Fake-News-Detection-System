import streamlit as st
import joblib

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detection System")

news = st.text_area(
    "Enter News Article",
    height=250
)

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some text.")
    else:

        x = vectorizer.transform([news])

        pred = model.predict(x)

        prob = model.predict_proba(x)

        confidence = max(prob[0]) * 100

        if pred[0] == 1:
            st.success(
                f"✅ REAL NEWS\nConfidence : {confidence:.2f}%"
            )
        else:
            st.error(
                f"❌ FAKE NEWS\nConfidence : {confidence:.2f}%"
            )