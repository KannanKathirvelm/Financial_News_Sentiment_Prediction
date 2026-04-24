"""
evaluation.py - Plotting utilities: confusion matrices, classification reports, training curves, model comparison.
"""
from __future__ import annotations
from typing import Dict,List,Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from .data_loader import LABEL_NAMES

def plot_confusion_matrix(cm,title="Confusion Matrix",savepath=None):
    cls=[LABEL_NAMES[i] for i in range(3)]; fig,ax=plt.subplots(figsize=(4.5,4))
    im=ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(cls); ax.set_yticklabels(cls); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    vmax=cm.max()
    for i in range(3):
        for j in range(3): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>vmax/2 else "black")
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); fig.tight_layout()
    if savepath: fig.savefig(savepath,dpi=150,bbox_inches="tight")
    return fig

def classification_table(y_true,y_pred):
    return pd.DataFrame(classification_report(y_true,y_pred,labels=[0,1,2],target_names=[LABEL_NAMES[i] for i in range(3)],output_dict=True,zero_division=0)).transpose().round(4)

def plot_training_curves(history,title_prefix="",savepath=None):
    epochs=range(1,len(history.train_loss)+1); fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3.5))
    ax1.plot(epochs,history.train_loss,label="train"); ax1.plot(epochs,history.val_loss,label="val"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title(f"{title_prefix} loss"); ax1.legend()
    ax2.plot(epochs,history.val_f1,label="val macro-F1",color="green"); ax2.plot(epochs,history.val_acc,label="val accuracy",color="orange"); ax2.set_xlabel("epoch"); ax2.set_ylabel("score"); ax2.set_title(f"{title_prefix} metrics"); ax2.legend()
    fig.tight_layout()
    if savepath: fig.savefig(savepath,dpi=150,bbox_inches="tight")
    return fig

def comparison_bar(results,metric="val_f1",savepath=None):
    names=list(results.keys()); values=[results[n][metric] for n in names]
    fig,ax=plt.subplots(figsize=(6,3.5)); bars=ax.bar(names,values,color=["#4C72B0","#55A868","#C44E52","#8172B2"][:len(names)])
    ax.set_ylabel(metric); ax.set_title(f"Model comparison — {metric}"); ax.set_ylim(0,max(values)*1.15)
    for bar,v in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,v+0.005,f"{v:.3f}",ha="center",va="bottom")
    fig.tight_layout()
    if savepath: fig.savefig(savepath,dpi=150,bbox_inches="tight")
    return fig

def summary_table(results):
    return pd.DataFrame([{"model":n,"accuracy":round(r["val_acc"],4),"macro_f1":round(r["val_f1"],4)} for n,r in results.items()]).sort_values("macro_f1",ascending=False).reset_index(drop=True)
