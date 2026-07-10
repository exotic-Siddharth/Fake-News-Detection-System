import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

text = input("Enter News : ")

x = vectorizer.transform([text])

pred = model.predict(x)

if pred[0]==1:
    print("REAL NEWS")
else:
    print("FAKE NEWS")