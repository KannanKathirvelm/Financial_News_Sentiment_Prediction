"""
models.py - RNN-family classifiers: SimpleRNN, LSTM, GRU for 3-class sentiment.
All return raw logits (batch, 3). Use CrossEntropyLoss during training.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class BaseTextClassifier(nn.Module):
    def __init__(self,vocab_size,embed_dim=128,hidden_dim=128,num_classes=3,pad_idx=0,dropout=0.3):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=pad_idx)
        self.dropout=nn.Dropout(dropout); self.pad_idx=pad_idx; self.hidden_dim=hidden_dim
    def masked_mean(self,hidden,x):
        mask=(x!=self.pad_idx).unsqueeze(-1).float()
        return (hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1.0)

class SimpleRNNClassifier(BaseTextClassifier):
    def __init__(self,vocab_size,embed_dim=128,hidden_dim=128,num_classes=3,pad_idx=0,dropout=0.3):
        super().__init__(vocab_size,embed_dim,hidden_dim,num_classes,pad_idx,dropout)
        self.rnn=nn.RNN(embed_dim,hidden_dim,batch_first=True); self.fc=nn.Linear(hidden_dim,num_classes)
    def forward(self,x):
        emb=self.dropout(self.embedding(x)); out,_=self.rnn(emb); return self.fc(self.dropout(self.masked_mean(out,x)))

class LSTMClassifier(BaseTextClassifier):
    def __init__(self,vocab_size,embed_dim=128,hidden_dim=128,num_classes=3,pad_idx=0,dropout=0.3):
        super().__init__(vocab_size,embed_dim,hidden_dim,num_classes,pad_idx,dropout)
        self.lstm=nn.LSTM(embed_dim,hidden_dim,batch_first=True,bidirectional=True); self.fc=nn.Linear(hidden_dim*2,num_classes)
    def forward(self,x):
        emb=self.dropout(self.embedding(x)); out,_=self.lstm(emb); return self.fc(self.dropout(self.masked_mean(out,x)))

class GRUClassifier(BaseTextClassifier):
    def __init__(self,vocab_size,embed_dim=128,hidden_dim=128,num_classes=3,pad_idx=0,dropout=0.3):
        super().__init__(vocab_size,embed_dim,hidden_dim,num_classes,pad_idx,dropout)
        self.gru=nn.GRU(embed_dim,hidden_dim,batch_first=True,bidirectional=True); self.fc=nn.Linear(hidden_dim*2,num_classes)
    def forward(self,x):
        emb=self.dropout(self.embedding(x)); out,_=self.gru(emb); return self.fc(self.dropout(self.masked_mean(out,x)))

MODEL_REGISTRY={"rnn":SimpleRNNClassifier,"lstm":LSTMClassifier,"gru":GRUClassifier}
def build_model(name,vocab_size,**kwargs):
    name=name.lower()
    if name not in MODEL_REGISTRY: raise ValueError(f"Unknown model '{name}'")
    return MODEL_REGISTRY[name](vocab_size=vocab_size,**kwargs)
