from transformers import pipeline
def predict(text):
    pipe = pipeline("text-classification", model="./models/task1_best", tokenizer="bert-base-uncased", return_all_scores=False)
    return pipe(text)
if __name__=='__main__':
    print(predict("Stocks rally after positive earnings"))