# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt

# ---------------------------
# 🎨 PAGE CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="Fake News Detector 📰",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------
# 💫 CUSTOM PAGE BACKGROUND
# ---------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141E30, #243B55);
}
h1, h2, h3, h4 {
    color: #FFD700 !important;
}
.stTextArea label, .stTextInput label {
    color: #fff !important;
}
div.stButton > button {
    background-color: #00C9A7;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 18px;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    background-color: #FF512F;
    transform: scale(1.05);
}
.stProgress > div > div > div > div {
    background-color: #FFD700;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------
# 🧠 LOAD MODEL & TOKENIZER
# ---------------------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('tokenizer/')  # <- uses saved tokenizer folder
    from transformers import TFDistilBertModel
    from tensorflow.keras.utils import custom_object_scope
    with custom_object_scope({'TFDistilBertModel': TFDistilBertModel}):
        model = tf.keras.models.load_model("best_hybrid_model.h5", compile=False)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# ⚙️ PREPROCESS FUNCTION
# ---------------------------
def preprocess_text(text, tokenizer, max_len=72):
    encoded = tokenizer(
        [text],
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )
    return encoded['input_ids'], encoded['attention_mask']

# ---------------------------
# 🧩 SIDEBAR
# ---------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=100)
    st.title("🧠 Fake News Detector")
    st.markdown("Detect whether a news article is **Fake** or **Real** with AI ⚡")
    st.markdown("---")
    st.markdown("**Developed by:** Mohammed Farhaan Shaikh ✨")
    st.markdown("Model: DistilBERT + CNN + GRU")

# ---------------------------
# 🖥️ MAIN SECTION
# ---------------------------
st.title("📰 Fake News Detection System")
st.markdown("### 📝 Enter a news headline or paragraph below:")

user_input = st.text_area("", height=150, placeholder="Type or paste your news article here...")

if st.button("🚀 Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some news text to analyze!")
    else:
        with st.spinner("🤖 AI analyzing your news... please wait..."):
            input_ids, attention_mask = preprocess_text(user_input, tokenizer)
            prediction = model.predict({'input_ids': input_ids, 'attention_mask': attention_mask})
            prob = float(prediction[0][0])

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        if prob > 0.4:
            label = "REAL 🟢"
            confidence = prob * 100
            st.success(f"✅ The News Seems **REAL** (Confidence: {confidence:.2f}%)")
        else:
            label = "FAKE 🔴"
            confidence = (1 - prob) * 100
            st.error(f"🚨 The News Seems **FAKE** (Confidence: {confidence:.2f}%)")

        st.progress(confidence / 100)
        st.metric(label="🧭 Confidence Level", value=f"{confidence:.2f}%")

        # ---------------------------
        # 📊 MATPLOTLIB CONFIDENCE GRAPH
        # ---------------------------
        # 📈 CONFIDENCE LINE CHART
        st.markdown("### 📈 Confidence Visualization")

        labels = ['Fake', 'Real']
        values = [(1 - prob) * 100, prob * 100]
        colors = ['#F44336', '#4CAF50']

        fig, ax = plt.subplots(figsize=(6, 3))

        # Plot line
        ax.plot(labels, values, color='#FFD700', linewidth=1, marker='o', markersize=5, markerfacecolor='#00C9A7')

        # Fill area under the curve for aesthetics
        ax.fill_between(labels, values, color='#FFD700', alpha=0.2)

        # Annotate points
        for i, v in enumerate(values):
            ax.text(i, v + 3, f"{v:.1f}%", color='white', ha='center', fontsize=8, fontweight='bold')

        ax.set_ylim(0, 100)
        ax.set_ylabel("Confidence (%)", color='white', fontsize=12)
        ax.set_title("Confidence Comparison (Fake vs Real)", color='#FFD700', fontsize=14)
        ax.grid(alpha=0.3)

# Dark theme style
        fig.patch.set_facecolor("#1f4037")
        ax.set_facecolor("#1f4037")
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        st.pyplot(fig)


