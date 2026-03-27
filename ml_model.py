import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class MLResumeModel:
    def __init__(self):
        # 🔥 Big improvement: n-grams
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.model = LogisticRegression()
        self.is_trained = False

    # 🔥 NEW: text cleaning
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        return text

    def train(self):
        df = pd.read_csv("training_data.csv")

        # 🔥 Clean text
        df["resume"] = df["resume"].apply(self.clean_text)
        df["job"] = df["job"].apply(self.clean_text)

        df["text"] = df["resume"] + " " + df["job"]

        X = self.vectorizer.fit_transform(df["text"])
        y = df["label"]

        self.model.fit(X, y)
        self.is_trained = True

    def predict_score(self, resume_text, job_desc):
        if not self.is_trained:
            self.train()

        # 🔥 Clean input
        resume_text = self.clean_text(resume_text)
        job_desc = self.clean_text(job_desc)

        text = resume_text + " " + job_desc
        X = self.vectorizer.transform([text])

        prob = self.model.predict_proba(X)[0][1]
        return round(prob * 100, 2)