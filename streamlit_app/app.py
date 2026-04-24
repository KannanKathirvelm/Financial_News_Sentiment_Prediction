"""
streamlit_app/app.py - Interactive sentiment dashboard.
Run: streamlit run streamlit_app/app.py
"""
from __future__ import annotations
import pickle,sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import LABEL_NAMES
from predict_sentiment import predict,MODELS_DIR

st.set_page_config(page_title="Financial Sentiment",page_icon="💹",layout="wide")
st.title("💹 Financial News Sentiment Classifier")
st.caption("Classify finance tweets as **Bearish**, **Bullish** or **Neutral** using RNN/LSTM/GRU or fine-tuned BERT.")

def _available_models():
    av=[]
    for m in ["rnn","lstm","gru"]:
        if (MODELS_DIR/f"{m}_best.pt").exists() and (MODELS_DIR/"vocab.pkl").exists(): av.append(m)
    if (MODELS_DIR/"bert_final").exists(): av.append("bert")
    return av

available=_available_models()
st.sidebar.header("⚙️ Settings")
if not available:
    st.sidebar.error("No trained models found. Run the notebook first."); st.stop()

model_choice=st.sidebar.selectbox("Model",options=available,index=len(available)-1)
st.sidebar.markdown("---")
st.sidebar.markdown("**Labels**\n\n- 🔴 **Bearish** — negative\n- 🟢 **Bullish** — positive\n- ⚪ **Neutral** — mixed")

col_left,col_right=st.columns([3,2])
with col_left:
    st.subheader("Input")
    user_text=st.text_area("Tweet / headline / paragraph","Apple beats earnings expectations, stock surges",height=140)
    go=st.button("Classify",type="primary",use_container_width=True)

with col_right:
    st.subheader("Prediction")
    if go and user_text.strip():
        with st.spinner(f"Running {model_choice.upper()}..."):
            try: result=predict(user_text.strip(),model_choice)
            except Exception as e: st.error(f"Error: {e}"); st.stop()
        emoji={"Bearish":"🔴","Bullish":"🟢","Neutral":"⚪"}[result["prediction"]]
        st.markdown(f"### {emoji} **{result['prediction']}**")
        probs=result["probabilities"]
        st.bar_chart(pd.DataFrame({"class":list(probs.keys()),"probability":list(probs.values())}).set_index("class"))
        st.dataframe(pd.DataFrame({"class":list(probs.keys()),"probability":[round(v,4) for v in probs.values()]}),use_container_width=True,hide_index=True)
    else:
        st.info("Type something and click **Classify**.")

st.markdown("---"); st.subheader("Validation-set class distribution")
dist_path=MODELS_DIR/"val_distribution.pkl"
if dist_path.exists():
    with open(dist_path,"rb") as f: dist_df=pickle.load(f)
    st.bar_chart(dist_df.set_index("class")["count"])
else:
    st.caption("Run the training notebook to populate this chart.")

results_path=MODELS_DIR/"results_summary.pkl"
if results_path.exists():
    st.markdown("---"); st.subheader("Model comparison")
    with open(results_path,"rb") as f: st.dataframe(pickle.load(f),use_container_width=True,hide_index=True)
