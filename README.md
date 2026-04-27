# Financial News Sentiment Prediction using Deep Learning & BERT

Classify finance-related tweets as **Bearish**, **Bullish** or **Neutral** using RNN baselines (SimpleRNN/LSTM/GRU) and optional fine-tuned BERT (DistilBERT / FinBERT).

> **Domain:** Finance & FinTech | **Dataset:** zeroshot/twitter-financial-news-sentiment | 9,938 train / 2,486 val

---

## Project Structure

```
financial_sentiment/
├── src/
│   ├── data_loader.py      # Dataset loading, tweet cleaning, vocab, DataLoaders
│   ├── models.py           # SimpleRNN, LSTM, GRU classifiers (PyTorch)
│   ├── train.py            # Training loop, early stopping, class-weighted loss
│   ├── bert_finetune.py    # HuggingFace Trainer BERT fine-tuning
│   └── evaluation.py       # Plots, confusion matrices, classification reports
├── notebooks/financial_sentiment_pipeline.ipynb
├── streamlit_app/app.py    # Interactive dashboard
├── predict_sentiment.py    # CLI inference script
├── generate_report.py      # Generates 2-page PDF report
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Running

```bash
# Train (Jupyter)
jupyter notebook notebooks/financial_sentiment_pipeline.ipynb

# CLI inference (after training)
python predict_sentiment.py --text "Apple beats earnings, stock surges"
python predict_sentiment.py --model bert --text "Fed holds rates"

# Streamlit dashboard
streamlit run streamlit_app/app.py

# Generate PDF report
python generate_report.py
```

---

## Expected Results

| Model | Val Accuracy | Val Macro F1 |
|-------|-------------|--------------|
| SimpleRNN | ~0.66 | ~0.52 |
| LSTM (bidir) | ~0.71 | ~0.60 |
| GRU (bidir) | ~0.73 | ~0.62 |
| DistilBERT | ~0.83 | ~0.77 |
| FinBERT | ~0.87 | ~0.83 |

---

## Dataset

- Source: `zeroshot/twitter-financial-news-sentiment` on Hugging Face
- LABEL_0 = Bearish | LABEL_1 = Bullish | LABEL_2 = Neutral
- Class imbalance: ~15% Bearish, ~30% Bullish, ~55% Neutral

---

## References

- https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
- https://huggingface.co/yiyanghkust/finbert-tone
- https://docs.streamlit.io
