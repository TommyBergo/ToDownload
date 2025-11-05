import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# =====================================================
# Task 2.2.3.1 — Sentiment Classifier without Constraint
# =====================================================
def run_task_1(model, tokenizer, sst):
    id2label = ["very negative", "negative", "neutral", "positive", "very positive"]
    test_texts = sst["test"]["text"]
    test_labels = sst["test"]["label"]

    # Build prompt for each sentence
    def build_prompt(sentence: str) -> str:
        return (
            f'The sentence is: "{sentence}".\n'
            f"Which is the sentiment among: very positive, positive, neutral, negative, very negative?\n"
            f"Answer with exactly one of these labels."
        )

    # Generate answer from model
    def get_sentiment(sentence: str) -> str:
        prompt = build_prompt(sentence)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

        # Match output with labels
        for lbl in id2label:
            if lbl in decoded:
                return lbl
        return "neutral"  # fallback if unclear output

    preds, gold = [], []
    print("\nRunning Task 2.2.3.1 (Free prompting)...")

    for text, label in tqdm(zip(test_texts, test_labels), total=len(test_texts)):
        pred_label = get_sentiment(text)
        preds.append(id2label.index(pred_label))
        gold.append(label)

    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")

    print("\n===== Final Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}\n")

    # Show some qualitative examples
    print("Sample predictions:")
    for i in range(5):
        print(f"\nText: {test_texts[i]}")
        print(f"Gold: {id2label[test_labels[i]]}")
        print(f"Pred: {id2label[preds[i]]}")

# =====================================================
# Task 2.2.3.2 — Chat format + Outlines + Literal
# =====================================================
def run_task_2(model, tokenizer, sst):
    from typing import Literal
    from outlines import models as omodels
    from outlines.text import generate, Template
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm

    print("\nRunning Task 2.2.3.2 (Chat format with Outlines)...")

    # Define allowed sentiment labels
    SentimentCategory = Literal["very positive", "positive", "neutral", "negative", "very negative"]

    # Wrap Gemma model into Outlines-compatible object
    outlines_model = omodels.transformers(model, tokenizer)

    # Create chat-format prompt template
    prompt_template = (
        "<start_header_id>system<end_header_id>\n"
        "You are a helpful assistant.\n"
        "<start_header_id>user<end_header_id>\n"
        "Classify the sentiment of this sentence into one of 5 categories: "
        "very positive, positive, neutral, negative, very negative.\n"
        "Text: {{text}}\n"
        "<start_header_id>assistant<end_header_id>\n"
    )
    template = Template.from_string(prompt_template)

    # Constrain model output to one of the 5 labels
    generator = generate.text(outlines_model, SentimentCategory, max_tokens=5)

    # Dataset and label mapping
    test_texts = sst["test"]["text"]
    test_labels = sst["test"]["label"]
    id2label = ["very negative", "negative", "neutral", "positive", "very positive"]

    preds, gold = [], []
    for text, label in tqdm(zip(test_texts, test_labels), total=len(test_texts)):
        rendered_prompt = template(text=text)
        pred_label = generator(rendered_prompt)
        if isinstance(pred_label, list): 
            pred_label = pred_label[0]
        preds.append(id2label.index(pred_label))
        gold.append(label)

    # Evaluate results
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")

    print("\n===== Final Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}\n")

    # Show sample outputs
    print("Sample predictions:")
    for i in range(5):
        print(f"\nText: {test_texts[i]}")
        print(f"Gold: {id2label[test_labels[i]]}")
        print(f"Pred: {id2label[preds[i]]}")


# =====================================================
# Task 2.2.3.3 — Structured reasoning (JSON output)
# =====================================================
#Strucure is the same of tak 2
def run_task_3(model, tokenizer, sst):
    from typing import Literal
    from pydantic import BaseModel
    from outlines import models as omodels
    from outlines.text import generate, Template
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm

    print("\nRunning Task 2.2.3.3 (Structured reasoning with explanation)...")

    # Define structured output schema
    class SentimentOutput(BaseModel):
        reason: str
        label: Literal["very positive", "positive", "neutral", "negative", "very negative"]

    # Wrap Gemma into Outlines model
    outlines_model = omodels.transformers(model, tokenizer)

    # Chat-style prompt encouraging reasoning before classification
    prompt_template = (
        "<start_header_id>system<end_header_id>\n"
        "You are a helpful assistant.\n"
        "<start_header_id>user<end_header_id>\n"
        "Analyze the sentiment of this sentence with reasoning steps first, "
        "then classify it into 5 categories: very positive, positive, neutral, negative, very negative.\n"
        "Text: {{text}}\n"
        "<start_header_id>assistant<end_header_id>\n"
    )
    template = Template.from_string(prompt_template)

    # Use Outlines JSON generator constrained by SentimentOutput schema
    generator = generate.json(outlines_model, SentimentOutput, max_tokens=100)

    # Dataset setup
    test_texts = sst["test"]["text"]
    test_labels = sst["test"]["label"]
    id2label = ["very negative", "negative", "neutral", "positive", "very positive"]

    preds, gold = [], []
    explanations = []

    # Inference loop
    for text, label in tqdm(zip(test_texts, test_labels), total=len(test_texts)):
        rendered_prompt = template(text=text)
        pred_output = generator(rendered_prompt)

        # Sometimes Outlines returns a list of dicts
        if isinstance(pred_output, list):
            pred_output = pred_output[0]

        # Extract label and reason
        reason = pred_output["reason"]
        label_str = pred_output["label"]

        explanations.append(reason)
        preds.append(id2label.index(label_str))
        gold.append(label)

    # Evaluation
    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")

    print("\n===== Final Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}\n")

    # Show qualitative examples
    print("Sample predictions with explanations:")
    for i in range(5):
        print(f"\nText: {test_texts[i]}")
        print(f"Gold: {id2label[test_labels[i]]}")
        print(f"Pred: {id2label[preds[i]]}")
        print(f"Reason: {explanations[i]}")





# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Run SST-5 sentiment classification (Task 2.2.3.x).")
    parser.add_argument(
        "--task",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Select which subtask to run: 1 = Free prompting | 2 = Outlines Literal | 3 = Outlines + Pydantic JSON"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-it",
        help="Pretrained model name (default: google/gemma-3-1b-it)"
    )
    args = parser.parse_args()


    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    # Load dataset
    print("\nLoading SST-5 dataset...")
    sst = load_dataset("SetFit/sst5")

    # Select and run the task
    if args.task == 1:
        print("\nSelected Task 2.2.3.1 — Free Prompting (Gemma LLM without constraints)")
        run_task_1(model, tokenizer, sst)

    elif args.task == 2:
        print("\nSelected Task 2.2.3.2 — Chat format + Outlines + Literal constraint")
        run_task_2(model, tokenizer, sst)

    elif args.task == 3:
        print("\nSelected Task 2.2.3.3 — Structured reasoning (JSON output with explanations)")
        run_task_3(model, tokenizer, sst)



if __name__ == "__main__":
    main()
