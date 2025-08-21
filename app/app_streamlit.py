import streamlit as st
import os
from transformers import pipeline

st.title("News Topic Classifier (BERT)")
st.write("Enter a headline and get the predicted topic.")

text = st.text_area("Headline", "Stocks rally after positive earnings")

if st.button("Predict"):
    with st.spinner("Loading model..."):
        MODEL_PATH = "./models/task1_best"

        if os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH):
            # Load your fine-tuned local model
            pipe = pipeline("text-classification", model=MODEL_PATH)
        else:
            # Fallback to a pre-trained AG News model from Hugging Face
            pipe = pipeline(
                "text-classification",
                model="textattack/bert-base-uncased-ag-news"
            )

        out = pipe(text)[0]
        st.write("Prediction:", out)
