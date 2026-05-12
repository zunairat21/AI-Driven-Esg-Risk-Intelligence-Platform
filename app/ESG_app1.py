# === Threading Fixes (Must be FIRST!) ===
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LIGHTGBM_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# Automatically use GPU if available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch version:", torch.__version__)
torch.set_num_threads(1)

# === Streamlit App Setup ===
import streamlit as st
st.set_page_config(page_title="ESG Risk Analyzer", layout="wide")

# === Standard Imports ===
import pandas as pd
import numpy as np
from joblib import load
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from joblib import parallel_backend
from joblib import load
# === Load Finetuned Sentiment Model ===
model_path = "/Users/apple/Desktop/Python_notebooks1.0/ESG_Risk_Analyzer/finetuned_finbert_esg"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=device)

# === Load FLAN-T5 Generator ===
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5 base model
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda" if torch.cuda.is_available() else "cpu")
# === Load RAG Embeddings ===
from sentence_transformers import SentenceTransformer
import faiss
from joblib import load
import numpy as np

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Load FAISS index
faiss_index = faiss.read_index("esg_faiss.index")

# Load metadata
try:
    metadata = load("esg_metadata.pkl")
except FileNotFoundError:
    metadata = []
    print("⚠️ Warning: esg_metadata.pkl not found. Proceeding without metadata.")

# Optional: Load embeddings (only if needed elsewhere)
embeddings = load("esg_embeddings.pkl")

def generate_answer(query, top_k=1):
    # Step 1: Embed the query and search FAISS index
    query_embedding = embedding_model.encode([query])
    _, indices = faiss_index.search(np.array(query_embedding).astype("float32"), top_k)
    top_idx = indices[0][0]

    # Step 2: Retrieve article
    retrieved_article = metadata[top_idx]["content"]

    # Step 3: Format prompt for FLAN-T5
    prompt = f"""
You are an ESG news analysis assistant.

Use the article below to answer the user's question based only on the content.

Article:
\"\"\"
{retrieved_article}
\"\"\"

Question: {query}
Answer:
"""

    # Step 4: Tokenize and generate answer using FLAN-T5
    input_ids = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(flan_model.device)
    output_ids = flan_model.generate(
        input_ids, max_length=256, do_sample=True, top_k=50, top_p=0.95, temperature=0.7
    )
    answer = flan_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Step 5: Fallback answer if model returns empty or short response
    if len(answer.strip().split()) < 3:
        answer = "The article discusses ESG topics, but specific measures are not clearly mentioned."

    return retrieved_article, answer
# === Load Classification/Regression Models ===
lgb_cls = load('lgb_model.joblib')    # classification model
label_encoder = load('label_encoder.joblib')    # for decoding class index
preprocessor = load('preprocessor.joblib')      # to transform input
reg_pipeline = load('rf_regression_pipeline.joblib')    # regression model

# === Load Sentiment Model & Define Prediction Function ===

# Prediction function
def predict_sentiment(text):
    outputs = pipeline(text)
    scores = outputs[0]
    pred_label = max(scores, key=lambda x: x['score'])['label']
    return pred_label

# === Streamlit UI ===
st.title("📊 ESG Risk Analyzer")
st.markdown("""
Welcome to the **ESG Risk Analyzer App**.
- Predict **ESG Risk Level** (Classification)
- Estimate **ESG Risk Score** (Regression)
""")

# === Sidebar Inputs ===
st.sidebar.header("📋 Input Company Features")
def user_input():
    return pd.DataFrame([{
        "CompanyName": st.sidebar.text_input("Company Name", "ABC Corp"),
        "Industry": st.sidebar.selectbox("Industry", ['Finance', 'Technology', 'Healthcare', 'Energy', 'Utilities']),
        "Region": st.sidebar.selectbox("Region", ['North America', 'Europe', 'Asia', 'South America', 'Africa']),
        "MarketCap": st.sidebar.number_input("Market Cap (in millions)", 0.0, value=500.0),
        "Revenue": st.sidebar.number_input("Revenue (in millions)", 0.0, value=100.0),
        "ProfitMargin": st.sidebar.slider("Profit Margin (%)", 0.0, 100.0, 15.0),
        "GrowthRate": st.sidebar.slider("Growth Rate (%)", 0.0, 100.0, 10.0),
        "ESG_Environmental": st.sidebar.slider("ESG Environmental Score", 0.0, 100.0, 60.0),
        "ESG_Social": st.sidebar.slider("ESG Social Score", 0.0, 100.0, 65.0),
        "ESG_Governance": st.sidebar.slider("ESG Governance Score", 0.0, 100.0, 70.0),
    }])

input_df = user_input()
company_name = input_df['CompanyName'][0]

# === Prediction Buttons ===
st.subheader("🔎 Predictions")
col1, col2 = st.columns(2)

if col1.button("🔍 Predict ESG Risk Level (Classification)"):
    try:
        # Ensure exact column order used in training
        X_cls_cols = ['Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate',
                      'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
                      'Industry', 'Region']
        input_df_ordered = input_df[X_cls_cols]
        input_transformed = pd.DataFrame(
        preprocessor.transform(input_df_ordered),
        columns=preprocessor.get_feature_names_out()
        )

        with parallel_backend('threading', n_jobs=1):
            pred_class = lgb_cls.predict(input_transformed)

        risk_label = label_encoder.inverse_transform(pred_class)[0]
        st.session_state['risk_label'] = risk_label
        st.success(f"🛡️ {company_name}'s Predicted ESG Risk Level: **{risk_label}**")

    except Exception as e:
        st.error(f"⚠️ Classification failed: {e}")

if col2.button("📈 Predict ESG Risk Score (Regression)"):
    try:
        # Ensure same feature columns as used in training
        X_reg_cols = ['Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate',
                      'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
                      'Industry', 'Region']
        input_df_ordered = input_df[X_reg_cols]
        predicted_score = reg_pipeline.predict(input_df_ordered)[0]

        st.session_state['predicted_score'] = predicted_score
        st.info(f"📊 {company_name}'s Predicted ESG Risk Score: **{predicted_score:.2f} / 100**")

    except Exception as e:
        st.error(f"⚠️ Regression failed: {e}")

# === News Sentiment ===
st.markdown("---")
st.subheader("🗞️ ESG News Headline Sentiment (Single Entry)")
news_input = st.text_area("Enter a news headline or snippet", "")
if st.button("Analyze Sentiment"):
    if news_input.strip():
        sentiment = predict_sentiment(news_input)
        st.success(f"🧠 Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a valid news headline.")

# === FLAN-T5 QA ===
st.markdown("---")
st.subheader("🤖 ESG News Insights (FLAN-T5 Q&A)")
user_query = st.text_input("Ask a question about ESG topics or companies")

if st.button("Generate Answer") and user_query:
    with st.spinner("Generating answer using FLAN-T5..."):
        article, answer = generate_answer(user_query)
        st.markdown("**📄 Retrieved Article:**")
        st.write(article)
        st.markdown("**🧠 Answer:**")
        st.success(answer)

# SHAP Explainability Section
st.markdown("---")
st.subheader("🧠 Model Explainability using SHAP")

with st.expander("See SHAP Summary Plots"):
    tab1, tab2 = st.tabs(["📊 ESG Risk Level (Classification)", "📈 ESG Risk Score (Regression)"])

    with tab1:
        st.image("assets/shap_classification_summary.png", caption="SHAP Summary - Classification Model", use_container_width=True)
        st.markdown("""
        SHAP shows how features impact the ESG risk level.

        - 🔵 High Risk → Weak ESG alignment  
        - 🔴 Low Risk → Strong ESG standing  
        - 🟣 Medium Risk → Balanced ESG behavior  
        - 🟢 Very High Risk → Critical ESG concerns  
        """)

    with tab2:
        st.image("assets/shap_regression_summary.png", caption="SHAP Summary - Regression Model", use_container_width=True)
        st.markdown("""
        SHAP explains how each factor pushes ESG score up/down.

        - 🔴 Red → High positive contribution  
        - 🔵 Blue → Negative/low contribution  
        - 🟣 Purple → Medium contribution  
        """)

# Report Generation
st.markdown("---")
st.subheader("📄 Download ESG Risk Report")

def generate_report():
    explanation = f"""
    ESG Risk Report for: {company_name}

    • ESG Risk Score: {st.session_state['predicted_score']:.2f}/100
    • ESG Risk Level: {st.session_state['risk_label']}

    Key ESG Inputs:
    - Environmental: {input_df['ESG_Environmental'][0]}
    - Social: {input_df['ESG_Social'][0]}
    - Governance: {input_df['ESG_Governance'][0]}

    Interpretation:
    The ESG risk level is determined by analyzing the balance of Environmental, Social, and Governance scores.
    Strong scores reduce risk, while weak areas increase it.

    SHAP Insight:
    SHAP values show which features had the most influence on the prediction.
    Red bars mean strong positive effect (e.g., high environmental score pushing ESG score up).
    Blue bars show features reducing the ESG rating.

    Risk Levels:
    🔵 High Risk → Weak ESG alignment  
    🔴 Low Risk → Strong ESG  
    🟣 Medium Risk → Balanced  
    🟢 Very High Risk → Concerning ESG performance
    """
    return explanation

def plot_esg_bar():
    fig, ax = plt.subplots()
    scores = [
        input_df['ESG_Environmental'][0],
        input_df['ESG_Social'][0],
        input_df['ESG_Governance'][0]
    ]
    labels = ['Environmental', 'Social', 'Governance']
    colors = ['green', 'skyblue', 'orange']
    ax.bar(labels, scores, color=colors)
    ax.set_ylim(0, 100)
    ax.set_title(f"{company_name} - ESG Component Breakdown", color='white')
    ax.set_facecolor('#111111')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#111111')
    return fig

if 'predicted_score' in st.session_state and 'risk_label' in st.session_state:
    report_text = generate_report()

    # Text download
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="ESG_Report_{company_name}.txt">📥 Download ESG Risk Report (.txt)</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Chart download
    fig = plot_esg_bar()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode()
    href_img = f'<a href="data:image/png;base64,{b64_img}" download="ESG_Chart_{company_name}.png">📉 Download ESG Chart (.png)</a>'
    st.markdown(href_img, unsafe_allow_html=True)

# === About This App ===
st.markdown("---")
st.subheader("ℹ️ About This App")

st.markdown("""
The **ESG Risk Analyzer** is a capstone project built to evaluate Environmental, Social, and Governance (ESG) risk using:
- 📊 ESG Risk Level Classification
- 📈 ESG Score Prediction
- 🤖 ESG News Q&A using FLAN-T5
- 🧠 Explainable AI using SHAP
- 📄 Downloadable ESG Reports

**Built with**: Streamlit, Scikit-learn, Transformers, Sentence Transformers, SHAP  
**Author**: Zunaira Tariq  
""")
