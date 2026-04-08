import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("turkish_toxic_language.csv")

print(df["is_toxic"].value_counts())

X = df["text"]
Y = df["is_toxic"]

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=20000
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_vec, Y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

pred = model.predict(X_test)
score = model.score(X_test, Y_test)

print(classification_report(Y_test,pred))
print("Accuracy:", score)

while True:
    print("Enter a sentence to classify:")
    sentence = input()
    result = model.predict(vectorizer.transform([sentence]))[0]
    print("The sentence is classified as:", "toxic" if result == 1 else "not toxic")
