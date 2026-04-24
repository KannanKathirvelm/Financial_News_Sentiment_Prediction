"""
data_loader.py - Data loading and preprocessing for Twitter Financial News Sentiment.
Labels: 0->Bearish, 1->Bullish, 2->Neutral
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

LABEL_NAMES: Dict[int, str] = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
NAME_TO_LABEL: Dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}
PAD_TOKEN = "<PAD>"; UNK_TOKEN = "<UNK>"; PAD_IDX = 0; UNK_IDX = 1

_URL_RE = re.compile(r"http\S+|www\.\S+"); _MENTION_RE = re.compile(r"@\w+"); _WS_RE = re.compile(r"\s+"); _PUNCT_RE = re.compile(r"[^\w\s\$\.\%\-]")

def clean_tweet(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower(); t = _URL_RE.sub(" ", t); t = _MENTION_RE.sub(" ", t); t = _PUNCT_RE.sub(" ", t); t = _WS_RE.sub(" ", t).strip(); return t

def simple_tokenize(text: str) -> List[str]: return clean_tweet(text).split()

def load_hf_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    from datasets import load_dataset
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    return ds["train"].to_pandas(), ds["validation"].to_pandas()

@dataclass
class Vocabulary:
    stoi: Dict[str, int]; itos: List[str]
    @classmethod
    def build(cls, texts, min_freq=2, max_size=30000):
        counter = Counter()
        for t in texts: counter.update(simple_tokenize(t))
        most_common = [w for w,c in counter.most_common() if c>=min_freq][:max_size-2]
        itos = [PAD_TOKEN, UNK_TOKEN] + most_common
        return cls(stoi={w:i for i,w in enumerate(itos)}, itos=itos)
    def __len__(self): return len(self.itos)
    def encode(self, text, max_len):
        ids = [self.stoi.get(t, UNK_IDX) for t in simple_tokenize(text)]
        return (ids[:max_len] if len(ids)>=max_len else ids+[PAD_IDX]*(max_len-len(ids)))

class TweetDataset(Dataset):
    def __init__(self, df, vocab, max_len=48):
        self.texts=df["text"].astype(str).tolist(); self.labels=df["label"].astype(int).tolist(); self.vocab=vocab; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return torch.tensor(self.vocab.encode(self.texts[idx],self.max_len),dtype=torch.long), torch.tensor(self.labels[idx],dtype=torch.long)

def make_dataloaders(train_df, val_df, vocab, batch_size=64, max_len=48):
    from torch.utils.data import DataLoader
    return DataLoader(TweetDataset(train_df,vocab,max_len),batch_size=batch_size,shuffle=True), DataLoader(TweetDataset(val_df,vocab,max_len),batch_size=batch_size,shuffle=False)

def describe_split(df, name="split"):
    counts=df["label"].value_counts().sort_index(); dist=pd.DataFrame({"label":counts.index,"class":[LABEL_NAMES[i] for i in counts.index],"count":counts.values,"pct":(counts.values/len(df)*100).round(2)})
    print(f"=== {name} — {len(df)} rows ==="); print(dist.to_string(index=False)); return dist

def compute_class_weights(labels, num_classes=3):
    counts=np.bincount(labels,minlength=num_classes).astype(float); counts=np.where(counts==0,1.0,counts)
    return torch.tensor(counts.sum()/(num_classes*counts),dtype=torch.float32)
