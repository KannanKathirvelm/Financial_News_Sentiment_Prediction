"""
train.py - Training loop with early stopping, class-weighted loss, gradient clipping.
"""
from __future__ import annotations
import copy,time
from dataclasses import dataclass,field
from typing import Dict,List,Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from torch.utils.data import DataLoader

@dataclass
class TrainHistory:
    train_loss:List[float]=field(default_factory=list)
    val_loss:List[float]=field(default_factory=list)
    val_acc:List[float]=field(default_factory=list)
    val_f1:List[float]=field(default_factory=list)

def compute_class_weights(labels,num_classes=3):
    counts=np.bincount(labels,minlength=num_classes).astype(float)
    counts=np.where(counts==0,1.0,counts)
    return torch.tensor(counts.sum()/(num_classes*counts),dtype=torch.float32)

@torch.no_grad()
def evaluate(model,loader,device,criterion=None):
    model.eval(); preds=[]; gold=[]; total_loss=total_n=0
    for x,y in loader:
        x,y=x.to(device),y.to(device); logits=model(x)
        if criterion is not None:
            total_loss+=criterion(logits,y).item()*x.size(0); total_n+=x.size(0)
        preds.extend(logits.argmax(-1).cpu().tolist()); gold.extend(y.cpu().tolist())
    acc=accuracy_score(gold,preds); f1=f1_score(gold,preds,average="macro")
    cm=confusion_matrix(gold,preds,labels=[0,1,2])
    return total_loss/max(total_n,1),acc,f1,np.array(gold),np.array(preds),cm

def train_model(model,train_loader,val_loader,*,epochs=8,lr=1e-3,class_weights=None,patience=3,device=None,verbose=True):
    device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion=nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    history=TrainHistory(); best_f1=-1.0; best_state=None; best_stats={}; no_improve=0
    for epoch in range(1,epochs+1):
        t0=time.time(); model.train(); rl=rn=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(); loss=criterion(model(x),y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            rl+=loss.item()*x.size(0); rn+=x.size(0)
        tl=rl/rn; vl,va,vf,yt,yp,cm=evaluate(model,val_loader,device,criterion)
        history.train_loss.append(tl); history.val_loss.append(vl); history.val_acc.append(va); history.val_f1.append(vf)
        if verbose: print(f"epoch {epoch:02d}  train={tl:.4f}  val={vl:.4f}  acc={va:.4f}  f1={vf:.4f}  ({time.time()-t0:.1f}s)")
        if vf>best_f1:
            best_f1=vf; best_state=copy.deepcopy(model.state_dict())
            best_stats={"val_loss":vl,"val_acc":va,"val_f1":vf,"confusion_matrix":cm,"y_true":yt,"y_pred":yp,"epoch":epoch}; no_improve=0
        else:
            no_improve+=1
            if no_improve>=patience:
                if verbose: print(f"Early stop at epoch {epoch}"); break
    if best_state: model.load_state_dict(best_state)
    return model,history,best_stats

def predict_text(text,model,vocab,max_len=48,device=None):
    device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    ids=torch.tensor([vocab.encode(text,max_len)],dtype=torch.long,device=device)
    with torch.no_grad():
        probs=F.softmax(model(ids),dim=-1).cpu().numpy()[0]
    return int(probs.argmax()),probs
