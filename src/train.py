import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1_macro": f1_score(p.label_ids, preds, average="macro")}

def main():
    ds = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=64)
    ds = ds.map(tok, batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    args = TrainingArguments(output_dir="./results/task1", evaluation_strategy="steps", eval_steps=500, logging_steps=200, per_device_train_batch_size=16, per_device_eval_batch_size=32, fp16=False, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="f1_macro")
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"], compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model("./models/task1_best")
if __name__=='__main__':
    main()
