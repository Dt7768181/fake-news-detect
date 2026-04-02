from flask import Flask, request, jsonify
from flask_cors import CORS
import re, emoji, joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, static_folder="static")
CORS(app)

CALL_TO_ACTION_PHRASES = [
    "details here", "read more", "full report",
    "click here", "see more", "learn more"
]

LABEL_MAP = {0: "Real News", 1: "Fake News"}

# ---------------- LOAD MODELS ---------------- #

svm_model = joblib.load("svm_model.pkl")
lgbm_model = joblib.load("lightgbm_model.pkl")

svm_vectorizer = joblib.load("tfidf_vectorizer.pkl")
lgbm_vectorizer = joblib.load("tfidf_vectorizer_lgbm.pkl")

# ---------------- PREPROCESSING ---------------- #

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)

    for phrase in CALL_TO_ACTION_PHRASES:
        text = text.replace(phrase, "")

    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------- FEATURE HELPERS ---------------- #

def extract_keywords(text, vectorizer, top_n=6):
    X = vectorizer.transform([text])
    scores = X.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()

    sorted_idx = np.argsort(scores)[::-1]
    keywords = []

    for i in sorted_idx:
        if scores[i] == 0:
            continue

        word = feature_names[i]
        if not any(word in k or k in word for k in keywords):
            keywords.append(word)

        if len(keywords) == top_n:
            break

    return keywords


def get_tfidf_weights(X_df, vectorizer, top_n=6):
    scores = X_df.values[0]
    feature_names = vectorizer.get_feature_names_out()
    idx = np.argsort(scores)[-top_n:]

    return {
        feature_names[i]: round(float(scores[i]), 3)
        for i in reversed(idx) if scores[i] > 0
    }


def get_lgbm_weights(X_df, vectorizer, top_n=6):
    tfidf_scores = X_df.values[0]
    importances = lgbm_model.feature_importances_
    combined = tfidf_scores * importances

    if combined.max() == 0:
        return {}

    idx = np.argsort(combined)[-top_n:]
    feature_names = vectorizer.get_feature_names_out()

    return {
        feature_names[i]: round(float(combined[i] / combined.max()), 3)
        for i in reversed(idx) if combined[i] > 0
    }

# ---------------- ENSEMBLE ---------------- #

def ensemble_classical(svm_p, svm_c, lgbm_p, lgbm_c):

    # 🔥 Bias correction: both say fake but not super confident
    if svm_p == 1 and lgbm_p == 1:
        if svm_c < 0.6 and lgbm_c < 0.95:
            return 0, "Low confidence override (likely real)"

    # Agreement → trust
    if svm_p == lgbm_p:
        return svm_p, "Agreement"

    # Prefer SVM (more stable)
    return svm_p, "SVM preferred (more stable)"

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    raw_text = data.get("text", "").strip()

    if not raw_text:
        return jsonify({"error": "Empty input"}), 400

    processed_text = preprocess(raw_text)

    # ---------------- VECTORIZE ---------------- #

    X_svm = svm_vectorizer.transform([processed_text])
    X_svm_df = pd.DataFrame(
        X_svm.toarray(),
        columns=svm_vectorizer.get_feature_names_out()
    )

    X_lgbm = lgbm_vectorizer.transform([processed_text])
    X_lgbm_df = pd.DataFrame(
        X_lgbm.toarray(),
        columns=lgbm_vectorizer.get_feature_names_out()
    )

    # ---------------- SVM ---------------- #

    svm_pred = int(svm_model.predict(X_svm_df)[0])
    svm_score = svm_model.decision_function(X_svm_df)[0]
    svm_conf = float(1 / (1 + np.exp(-abs(svm_score))))

    # ---------------- LGBM ---------------- #

    lgbm_pred = int(lgbm_model.predict(X_lgbm_df)[0])
    lgbm_conf = float(np.max(lgbm_model.predict_proba(X_lgbm_df)))

    # ---------------- DEBUG ---------------- #

    print("SVM:", svm_pred, svm_conf)
    print("LGBM:", lgbm_pred, lgbm_conf)

    # ---------------- FINAL DECISION ---------------- #

    final_pred, reason = ensemble_classical(
        svm_pred, svm_conf,
        lgbm_pred, lgbm_conf
    )

    return jsonify({
        "final_prediction": LABEL_MAP[final_pred],
        "decision_reason": reason,

        "svm": {
            "pred": LABEL_MAP[svm_pred],
            "confidence": round(svm_conf, 3),
            "weights": get_tfidf_weights(X_svm_df, svm_vectorizer)
        },

        "lightgbm": {
            "pred": LABEL_MAP[lgbm_pred],
            "confidence": round(lgbm_conf, 3),
            "weights": get_lgbm_weights(X_lgbm_df, lgbm_vectorizer)
        },

        "bert": None,
        "keywords": extract_keywords(processed_text, svm_vectorizer)
    })


# ---------------- STARTUP ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)