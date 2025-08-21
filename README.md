# Task 1 — News Topic Classifier (BERT)

## 🎯 Objective
Fine-tune a transformer model (`bert-base-uncased`) on the AG News dataset to classify news headlines into topic categories.

## 🛠️ Methodology / Approach
- Loaded the **AG News dataset** from Hugging Face.  
- Preprocessed and tokenized the text using Hugging Face `transformers`.  
- Fine-tuned the `bert-base-uncased` model for multi-class classification.  
- Evaluated performance using **Accuracy** and **F1-score**.  
- Deployed the model with **Streamlit/Gradio** for live predictions.  

## 📊 Key Results / Observations
- Achieved strong classification accuracy (~94–95%).  
- F1-scores across classes show consistent performance.  
- Deployment allows real-time headline classification into categories like *World, Sports, Business, Sci/Tech*.  


See `notebooks/`, `src/`, and `app/`.
