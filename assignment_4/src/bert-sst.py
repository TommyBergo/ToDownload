import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def main():
    # ---------------------------
    # 1. Command-line arguments
    # ---------------------------
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained BERT model on SST-5 sentiment classification.")
    parser.add_argument("--model_name", type=str, default="bert-base-cased",
                        help="Name of the pretrained model (e.g., 'bert-base-cased', 'roberta-base').")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (L2 regularization).")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    args = parser.parse_args()

    # ---------------------------
    # 2. Dataset loading
    # ---------------------------
    print("\nLoading SST-5 dataset...")
    sst = load_dataset("SetFit/sst5")

    # ---------------------------
    # 3. Tokenization
    # ---------------------------
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_length)

    tokenized_sst = sst.map(tokenize_function, batched=True)
    tokenized_sst = tokenized_sst.rename_column("label", "labels")
    tokenized_sst.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # ---------------------------
    # 4. Model setup
    # ---------------------------
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)

    # ---------------------------
    # 5. Metrics
    # ---------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1": f1}

    # ---------------------------
    # 6. Training parameters
    # ---------------------------
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        report_to="none",    # sobstitute do report_to="wandb" to get the graphics !!
        load_best_model_at_end=True
    )

    # ---------------------------
    # 7. Trainer setup
    # ---------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_sst["train"],
        eval_dataset=tokenized_sst["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---------------------------
    # 8. Training
    # ---------------------------
    print("\nStarting training...\n")
    trainer.train()

    # ---------------------------
    # 9. Final evaluation
    # ---------------------------
    print("\n===== Final Evaluation =====")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # ---------------------------
    # 10. Classification report & confusion matrix
    # ---------------------------
    preds = trainer.predict(tokenized_sst["test"]).predictions
    preds = np.argmax(preds, axis=-1)
    labels = np.array(tokenized_sst["test"]["labels"])

    print("\nClassification Report:")
    print(classification_report(
        labels,
        preds,
        target_names=["very negative", "negative", "neutral", "positive", "very positive"],
        digits=4
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()



