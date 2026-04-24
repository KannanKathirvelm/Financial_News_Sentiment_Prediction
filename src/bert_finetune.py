"""
bert_finetune.py - Optional BERT fine-tuning via HuggingFace Trainer.
Default: distilbert-base-uncased. Swap for yiyanghkust/finbert-tone for finance-specific pretraining.
"""
from __future__ import annotations
from typing import Dict,Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

DEFAULT_MODEL_NAME="distilbert-base-uncased"

def _compute_metrics(eval_pred):
    logits,labels=eval_pred; preds=np.argmax(logits,axis=-1)
    return {"accuracy":accuracy_score(labels,preds),"macro_f1":f1_score(labels,preds,average="macro")}

def finetune_bert(train_df,val_df,model_name=DEFAULT_MODEL_NAME,output_dir="./models/bert",epochs=3,batch_size=16,lr=2e-5,max_len=64):
    from transformers import AutoModelForSequenceClassification,AutoTokenizer,Trainer,TrainingArguments,DataCollatorWithPadding
    from datasets import Dataset
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3,id2label={0:"Bearish",1:"Bullish",2:"Neutral"},label2id={"Bearish":0,"Bullish":1,"Neutral":2})
    def tok(b): return tokenizer(b["text"],truncation=True,padding="max_length",max_length=max_len)
    train_ds=Dataset.from_pandas(train_df[["text","label"]],preserve_index=False).map(tok,batched=True)
    val_ds=Dataset.from_pandas(val_df[["text","label"]],preserve_index=False).map(tok,batched=True)
    args=TrainingArguments(output_dir=output_dir,num_train_epochs=epochs,per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size*2,learning_rate=lr,weight_decay=0.01,eval_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="macro_f1",report_to="none")
    trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,tokenizer=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=_compute_metrics)
    trainer.train()
    preds=trainer.predict(val_ds); y_pred=np.argmax(preds.predictions,axis=-1); labels=preds.label_ids
    return model,tokenizer,{"val_acc":accuracy_score(labels,y_pred),"val_f1":f1_score(labels,y_pred,average="macro"),"confusion_matrix":confusion_matrix(labels,y_pred,labels=[0,1,2]),"y_true":labels,"y_pred":y_pred}

def predict_bert(text,model,tokenizer,max_len=64):
    import torch
    model.eval(); device=next(model.parameters()).device
    enc=tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=max_len)
    enc={k:v.to(device) for k,v in enc.items()}
    with torch.no_grad(): probs=torch.softmax(model(**enc).logits,dim=-1).cpu().numpy()[0]
    return int(probs.argmax()),probs
