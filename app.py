import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle  # JSON yerine pickle kullanıyoruz
import time
import altair as alt
import random

# --- PAGE SETTINGS ---
st.set_page_config(
    page_title="Bloom AI | Sentiment Analysis",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- ROSE GOLD THEME ---
st.markdown("""
<style>
    .stApp { background-color: #FFF5F7; }
    h1 { color: #C75B7A; font-family: 'Helvetica Neue', sans-serif; font-weight: 300; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: white; border-radius: 20px; color: #C75B7A; border: 1px solid #F8C8DC; }
    .stTabs [aria-selected="true"] { background-color: #C75B7A; color: white; }
    .stTextArea textarea { background-color: white; border: 2px solid #F8C8DC; border-radius: 15px; }
    .stTextArea textarea:focus { border-color: #C75B7A; box-shadow: 0 0 10px rgba(199,91,122,0.3); }
    div.stButton > button { background: linear-gradient(135deg, #D4889E, #C75B7A); color: white; border-radius: 50px; border:none; padding: 10px 30px; width: 100%; font-weight: bold;}
    div.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(199,91,122,0.4); }
    .result-card { background: white; border: 2px solid #F8C8DC; border-radius: 20px; padding: 30px; text-align: center; margin-top: 20px; box-shadow: 0 4px 15px rgba(199,91,122,0.1); }
    .big-score { font-size: 5em; font-weight: 700; color: #C75B7A; margin: 0; }
    .status-text { font-size: 2em; font-weight: 600; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- MODEL LOADING (DÜZELTİLMİŞ) 🧠 ---
@st.cache_resource
def load_artifacts():
    try:
        # Model yükleme
        model = load_model("regression_lstm_yelp.h5", compile=False)
        
        # Tokenizer yükleme - PICKLE FORMATINDA
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        return model, tokenizer
    except FileNotFoundError as e:
        st.error(f"❌ Dosya bulunamadı: {e}")
        st.info("📌 Lütfen bu dosyaların aynı klasörde olduğundan emin ol: regression_lstm_yelp.h5 ve tokenizer.pkl")
        return None, None
    except Exception as e:
        st.error(f"Yükleme Hatası: {e}")
        return None, None

model, tokenizer = load_artifacts()

# --- DEMO DATA GENERATOR ---
def generate_demo_data():
    reviews = []
    dishes = ["Truffle Burger", "Spicy Ramen", "Margherita Pizza", "Grilled Salmon", "Caesar Salad", "Chocolate Souffle", "Latte", "Tacos", "Ribeye Steak", "Sushi Platter", "Cheesecake", "Fish & Chips", "Pad Thai"]
    times = ["last night", "today", "yesterday", "last weekend", "on Tuesday", "for lunch", "for dinner", "this morning"]
    partners = ["my husband", "my wife", "my boyfriend", "my girlfriend", "my colleagues", "my kids", "a friend", "my parents"]
    adj_pos = ["mind-blowing", "exquisite", "divine", "superb", "fresh", "juicy", "perfectly cooked", "flavorful", "fantastic", "mouth-watering"]
    adj_neg = ["inedible", "gross", "bland", "salty", "overcooked", "raw", "stale", "greasy", "cold", "frozen inside", "burnt", "tasteless"]
    staff_pos = ["attentive", "friendly", "knowledgeable", "sweet", "fast", "polite", "welcoming", "efficient"]
    staff_neg = ["rude", "dismissive", "slow", "unprofessional", "aggressive", "lazy", "arrogant", "absent"]

    for i in range(500):
        order_id = random.randint(10000, 99999) 
        rand_val = random.random()
        dish = random.choice(dishes)
        time_ref = random.choice(times)
        partner = random.choice(partners)
        text = ""
        if rand_val < 0.45:
            templates = [f"Order #{order_id}: Went there {time_ref} with {partner}. The {dish} was {random.choice(adj_pos)}!", f"Order #{order_id}: Absolutely loved the {dish}. The staff was {random.choice(staff_pos)} and kind."]
            text = random.choice(templates)
        elif rand_val < 0.80:
            templates = [f"Order #{order_id}: Worst experience {time_ref}. The {dish} was {random.choice(adj_neg)}.", f"Order #{order_id}: I will never go back. The waiter was {random.choice(staff_neg)} and ignored us."]
            text = random.choice(templates)
        else: 
            templates = [f"Order #{order_id}: The {dish} was okay, but the service was a bit {random.choice(staff_neg)}.", f"Order #{order_id}: Decent place for {dish}, but the music was too loud."]
            text = random.choice(templates)
        reviews.append(text)
    df = pd.DataFrame(reviews, columns=["text"])
    return df.to_csv(index=False).encode('utf-8')

# --- HEADER ---
st.markdown("<h1>🌸 Bloom AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B38896;'>Intelligent Sentiment Analytics for Businesses</p>", unsafe_allow_html=True)

# Model kontrol uyarısı
if model is None or tokenizer is None:
    st.error("⚠️ Model veya tokenizer yüklenemedi. Lütfen dosyaları kontrol edin.")
    st.stop()

# --- TABS ---
tab1, tab2 = st.tabs(["⚡ Quick Scan", "📂 Upload Excel"])

# --- TAB 1 ---
with tab1:
    user_input = st.text_area("Review", placeholder="Write a review to test... (e.g. The service was fantastic!)", height=150, label_visibility="collapsed")
    if st.button("ANALYZE REVIEW 🔮"):
        if not user_input:
            st.warning("Please enter a review.")
        else:
            with st.spinner('Thinking...'):
                time.sleep(0.5)
                seq = tokenizer.texts_to_sequences([user_input])
                pad = pad_sequences(seq, maxlen=100, padding="post", truncating="post")
                pred = model.predict(pad, verbose=0)[0][0]
                score = np.clip((pred * 4) + 1, 1, 5)
                
                if score >= 4.0: 
                    color, text, icon = "#68B984", "AMAZING", "😍"
                    st.balloons()
                elif score >= 2.5: 
                    color, text, icon = "#F7C04A", "AVERAGE", "🤔"
                else: 
                    color, text, icon = "#E76161", "BAD", "👎"
                
                st.markdown(f"""<div class="result-card"><div class="status-text" style="color:{color}">{text} {icon}</div><div class="big-score" style="color:{color}">{score:.1f}</div><p style="color:#aaa; margin-top:10px;">Confidence Score: {pred:.3f}</p></div>""", unsafe_allow_html=True)

# --- TAB 2 ---
with tab2:
    st.info("Don't have a file? Download our sample dataset below! 👇")
    demo_csv = generate_demo_data()
    st.download_button(label="📥 Download Sample Reviews (CSV)", data=demo_csv, file_name="bloom_ai_demo_data.csv", mime="text/csv")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload file containing a 'text' column", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'): 
                df = pd.read_csv(uploaded_file)
            else: 
                df = pd.read_excel(uploaded_file)
            
            text_col = "text" if "text" in df.columns else df.columns[0]
            
            if st.button("Analyze All Reviews 🚀"):
                progress_bar = st.progress(0)
                texts = df[text_col].astype(str).tolist()
                seqs = tokenizer.texts_to_sequences(texts)
                padded = pad_sequences(seqs, maxlen=100, padding="post", truncating="post")
                preds = model.predict(padded, verbose=0)
                final_scores = np.clip((preds * 4) + 1, 1, 5).flatten()
                df['AI_Score'] = final_scores
                df['Sentiment'] = pd.cut(df['AI_Score'], bins=[0, 2.5, 3.9, 6], labels=['Negative 👎', 'Neutral 🤔', 'Positive 😍'])
                progress_bar.progress(100)
                st.success("Analysis Complete! 🎉")
                
                avg_score = df['AI_Score'].mean()
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Reviews", len(df))
                col_b.metric("Average Score", f"{avg_score:.1f} / 5.0")
                col_c.metric("Happy Customers", len(df[df['AI_Score'] >= 4.0]))
                
                st.markdown("##### Sentiment Distribution")
                chart_data = df['Sentiment'].value_counts().reset_index()
                chart_data.columns = ['Sentiment', 'Count']
                c = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Sentiment', sort=None), 
                    y='Count', 
                    color=alt.value("#C75B7A"), 
                    tooltip=['Sentiment', 'Count']
                ).properties(height=300)
                st.altair_chart(c, use_container_width=True)
                
                st.markdown("### 🚨 Critical Reviews (Action Needed)")
                bad_reviews = df[df['AI_Score'] < 2.5][['AI_Score', text_col]].sort_values(by='AI_Score')
                st.dataframe(bad_reviews, use_container_width=True)
        except Exception as e: 
            st.error(f"Error reading file: {e}")