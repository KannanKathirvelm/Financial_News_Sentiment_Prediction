"""
predict_sentiment.py
--------------------
Automated CLI: takes a raw financial paragraph (or tweet) and prints the
predicted sentiment with class probabilities.

Usage
-----
    # Single sentence on command line
    python predict_sentiment.py --text "Apple beats earnings expectations, stock surges"

    # From a text file
    python predict_sentiment.py --file news.txt

    # Switch models
    python predict_sentiment.py --model bert --text "..."
    python predict_sentiment.py --model gru  --text "..."
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import LABEL_NAMES, Vocabulary
from src.models import build_model
from src.train import predict_text

MODELS_DIR = Path(__file__).parent / "models"


def load_rnn_model(model_name: str):
    ckpt_path = MODELS_DIR / f"{model_name}_best.pt"
    vocab_path = MODELS_DIR / "vocab.pkl"
    if not ckpt_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path} or {vocab_path}. Train the models first.")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    model = build_model(model_name, vocab_size=len(vocab))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model, vocab


def predict(text: str, model_name: str = "gru") -> dict:
    model_name = model_name.lower()
    if model_name == "bert":
        from src.bert_finetune import predict_bert
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        bert_dir = MODELS_DIR / "bert_final"
        tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
        label_id, probs = predict_bert(text, model, tokenizer)
    else:
        model, vocab = load_rnn_model(model_name)
        label_id, probs = predict_text(text, model, vocab)
    return {
        "text": text, "model": model_name,
        "prediction": LABEL_NAMES[label_id], "label_id": int(label_id),
        "probabilities": {LABEL_NAMES[0]: float(probs[0]), LABEL_NAMES[1]: float(probs[1]), LABEL_NAMES[2]: float(probs[2])},
    }


def main():
    parser = argparse.ArgumentParser(description="Predict financial tweet sentiment.")
    parser.add_argument("--text", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--model", type=str, default="gru", choices=["rnn","lstm","gru","bert"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    text = args.text or (Path(args.file).read_text().strip() if args.file else None)
    if not text:
        parser.error("Either --text or --file is required.")
    result = predict(text, args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nText       : {result['text']}")
        print(f"Prediction : {result['prediction']}")
        for cls, p in result["probabilities"].items():
            print(f"  {cls:<8}  {p:.4f}  {'#' * int(p*40)}")

if __name__ == "__main__":
    main()
