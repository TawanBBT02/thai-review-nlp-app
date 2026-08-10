import re
import pandas as pd
import streamlit as st
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.tag import pos_tag
from pythainlp.util import normalize

# --- CONFIGURATION ---
st.set_page_config(
    page_title="ระบบวิเคราะห์ข้อความรีวิวร้านอาหาร (TH/EN)",
    page_icon="🍔",
    layout="wide"
)

# --- CACHED NLP RESOURCES ---
@st.cache_data
def get_stopwords():
    """โหลด Stopwords ภาษาไทย และภาษาอังกฤษพื้นฐาน"""
    thai_sw = set(thai_stopwords())
    eng_sw = {
        "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "him", 
        "his", "she", "her", "it", "its", "they", "them", "their", "what", "which", 
        "who", "this", "that", "these", "those", "am", "is", "are", "was", "were", 
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "a", "an", 
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
        "at", "by", "for", "with", "about", "against", "between", "into", "through", 
        "during", "before", "after", "above", "below", "to", "from", "up", "down", 
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
    }
    return thai_sw.union(eng_sw)

@st.cache_data
def get_dictionaries():
    """โหลด Dictionary สำหรับ Topic และ Sentiment (รองรับ ไทย/อังกฤษ)"""
    topic_keywords = {
        "รสชาติ/อาหาร": [
            "อร่อย", "แซ่บ", "กลมกล่อม", "จาง", "เค็ม", "หวาน", "เผ็ด", "จาน", "เมนู", "กุ้ง", "เนื้อ", "หมู", "ซุป",
            "delicious", "tasty", "yummy", "food", "dish", "menu", "soup", "meat", "pork", "shrimp", "sweet", "salty", "spicy"
        ],
        "การบริการ": [
            "พนักงาน", "บริการ", "ต้อนรับ", "เสิร์ฟ", "พูดจา", "รอนาน", "ช้า", "เร็ว", "ใส่ใจ",
            "service", "staff", "waiter", "waitress", "serve", "slow", "fast", "quick", "polite", "rude"
        ],
        "ราคา/ความคุ้มค่า": [
            "ราคา", "แพง", "ถูก", "คุ้ม", "ปริมาณ", "บิล", "บาท", "จานใหญ่",
            "price", "expensive", "cheap", "worth", "value", "cost", "bill", "portion"
        ],
        "บรรยากาศ/สถานที่": [
            "ร้าน", "ที่นั่ง", "แอร์", "ที่จอดรถ", "สะอาด", "สกปรก", "มุมถ่ายรูป", "วิว", "บรรยากาศ",
            "place", "shop", "restaurant", "atmosphere", "ambiance", "clean", "dirty", "seat", "parking", "view"
        ]
    }

    sentiment_keywords = {
        "คำชม (Positive)": [
            "อร่อย", "ดี", "ชอบ", "ประทับใจ", "เร็ว", "สะอาด", "คุ้ม", "น่ารัก", "สด", "ยอดเยี่ยม",
            "good", "great", "excellent", "love", "like", "awesome", "amazing", "fresh", "clean", "fast", "cheap"
        ],
        "คำติ (Negative)": [
            "ช้า", "แพง", "ไม่อร่อย", "สกปรก", "รอนาน", "แย่", "เค็มเกิน", "ห่วย", "จาง", "เหม็น",
            "bad", "terrible", "worst", "slow", "dirty", "expensive", "salty", "disappointed", "poor", "horrible"
        ]
    }
    return topic_keywords, sentiment_keywords

# เรียกใช้งานข้อมูลที่ถูก Cache
STOPWORDS = get_stopwords()
TOPIC_KEYWORDS, SENTIMENT_KEYWORDS = get_dictionaries()

# --- NLP FUNCTIONS ---
def clean_text(text: str) -> str:
    """1. Regex & Cleansing: ลบ Noise, เบอร์โทร, URL, ปรับเป็นตัวพิมพ์เล็ก และลดคำลากเสียง"""
    if not isinstance(text, str):
        return ""
    
    # ลบ URL และ เบอร์โทรศัพท์
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'(\d{2,4}[-\s]?\d{3,4}[-\s]?\d{3,4})', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    
    # แปลงตัวอักษรภาษาอังกฤษเป็นตัวพิมพ์เล็ก
    text = text.lower()
    
    # ลดคำลากเสียง (เช่น อร่อยยยย -> อร่อย, goooood -> good)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Normalize สระภาษาไทย
    text = normalize(text)
    return text.strip()

def process_tokens(text: str):
    """2. Tokenization & Normalization: ตัดคำและลบ Stopwords ทั้ง TH/EN"""
    raw_tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    filtered_tokens = [w.strip() for w in raw_tokens if w.strip() not in STOPWORDS and len(w.strip()) > 1]
    return raw_tokens, filtered_tokens

def extract_pos_and_entities(tokens: list):
    """3. POS & Entity Extraction"""
    tagged = pos_tag(tokens, engine='perceptron', corpus='orchid')
    nouns = [word for word, tag in tagged if tag in ['NCMN', 'NPRE', 'NTNT'] or (word.isalnum() and word.isascii())]
    adjectives = [word for word, tag in tagged if tag in ['VATT', 'ADVN']]
    return tagged, list(set(nouns)), list(set(adjectives))

def classify_topics_and_sentiment(tokens: list):
    """4. Topic Identification & Sentiment Analysis"""
    detected_topics = []
    sentiments = []
    
    token_set = set(tokens)
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in token_set for kw in keywords):
            detected_topics.append(topic)
            
    for sentiment_type, keywords in SENTIMENT_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in token_set]
        if matched:
            sentiments.append(f"{sentiment_type}: ({', '.join(matched)})")
            
    return (
        detected_topics if detected_topics else ["ทั่วไป / General"],
        sentiments if sentiments else ["ไม่พบคำระบุความรู้สึกชัดเจน / Neutral"]
    )

# --- UI LAYOUT ---
st.title("🍔 ระบบคัดกรองและวิเคราะห์ข้อความรีวิว (TH/EN)")
st.subheader("Food & Product Review NLP Processing Application")

tab1, tab2 = st.tabs(["🔍 วิเคราะห์ข้อความเดี่ยว (Single Text)", "📂 วิเคราะห์ไฟล์ CSV (Batch Processing)"])

# --- TAB 1: SINGLE TEXT PROCESSING ---
with tab1:
    user_input = st.text_area(
        "กรอกข้อความรีวิวที่ต้องการวิเคราะห์ (รองรับไทยและอังกฤษ):",
        value="The food was very delicious and fresh! But the service was extremely slow. Call 081-234-5678 https://example.com",
        height=120
    )
    
    if st.button("ประมวลผลข้อความ", type="primary"):
        if user_input.strip():
            cleansed = clean_text(user_input)
            raw_tokens, filtered_tokens = process_tokens(cleansed)
            pos_tags, nouns, adjectives = extract_pos_and_entities(filtered_tokens)
            topics, sentiments = classify_topics_and_sentiment(filtered_tokens)
            
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🧹 1. ผลการ Cleansing (Regex)")
                st.info(cleansed if cleansed else "ข้อความว่างเปล่าหลัง Cleansing")
                
                st.markdown("### ✂️ 2. ผลการ Tokenization")
                st.write("**คำทั้งหมดที่ตัดได้:**", raw_tokens)
                st.write("**หลังลบ Stopwords (TH/EN):**", filtered_tokens)

            with col2:
                st.markdown("### 🏷️ 3. การจำแนกหัวข้อและ Sentiment")
                st.write("**หัวข้อที่เกี่ยวข้อง (Topics):**")
                for t in topics:
                    st.success(f"- {t}")
                st.write("**คำชม / คำติ ที่พบ:**")
                for s in sentiments:
                    st.warning(f"- {s}")

                st.markdown("### 📌 4. Key Elements (POS Tagging)")
                st.write("**คำนาม/เมนู/สถานที่/คำอังกฤษ (Nouns/Entities):**", nouns)
                st.write("**คำบรรยาย/คุณลักษณะ (Adjectives):**", adjectives)

# --- TAB 2: BATCH CSV PROCESSING ---
with tab2:
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV (ต้องมีคอลัมน์ชื่อ 'review_text')", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ CSV: {e}")
            df = None
            
        if df is not None:
            if df.empty:
                st.warning("ไฟล์ CSV ที่อัปโหลดไม่มีข้อมูล")
            elif "review_text" in df.columns:
                st.write("ตัวอย่างข้อมูลที่อัปโหลด (3 รายการแรก):", df.head(3))
                
                if st.button("ประมวลผลทั้งไฟล์", type="primary"):
                    results = []
                    for text in df["review_text"]:
                        cleaned = clean_text(str(text))
                        _, filtered = process_tokens(cleaned)
                        topics, sentiments = classify_topics_and_sentiment(filtered)
                        results.append({
                            "ข้อความเดิม": text,
                            "ข้อความหลังคลีน": cleaned,
                            "หัวข้อ": ", ".join(topics),
                            "ข้อสังเกต Sentiment": " | ".join(sentiments)
                        })
                    
                    result_df = pd.DataFrame(results)
                    st.success("ประมวลผลเสร็จสิ้น!")
                    
                    # --- DASHBOARD & VISUALIZATION ---
                    st.divider()
                    st.markdown("## 📊 สรุปภาพรวมการวิเคราะห์ (Dashboard)")
                    
                    # Metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("จำนวนรีวิวทั้งหมด", f"{len(result_df)} รายการ")
                    
                    all_topics = [t.strip() for sublist in result_df['หัวข้อ'].str.split(',') for t in sublist if t.strip()]
                    topic_counts = pd.Series(all_topics).value_counts()
                    top_topic = topic_counts.index[0] if len(topic_counts) > 0 else "-"
                    m2.metric("หัวข้อที่พบมากที่สุด", top_topic)
                    
                    pos_count = result_df['ข้อสังเกต Sentiment'].str.contains('คำชม').sum()
                    neg_count = result_df['ข้อสังเกต Sentiment'].str.contains('คำติ').sum()
                    m3.metric("สัดส่วนรีวิวที่มีคำชม / คำติ", f"{pos_count} / {neg_count}")

                    # Charts
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.markdown("### 📈 จำนวนรีวิวจำแนกตามหัวข้อ")
                        st.bar_chart(topic_counts)
                        
                    with chart_col2:
                        st.markdown("### ⚖️ เปรียบเทียบจำนวนคำชมและคำติที่พบ")
                        sentiment_summary = pd.DataFrame({
                            "ประเภท": ["คำชม (Positive)", "คำติ (Negative)"],
                            "จำนวน (ครั้ง)": [pos_count, neg_count]
                        }).set_index("ประเภท")
                        st.bar_chart(sentiment_summary)

                    # Data Table & Export
                    st.markdown("### 📋 ตารางข้อมูลผลการวิเคราะห์")
                    st.dataframe(result_df, use_container_width=True)
                    
                    csv_data = result_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 ดาวน์โหลดผลการวิเคราะห์ (CSV)", csv_data, "nlp_analysis_result.csv", "text/csv")
            else:
                st.error("ไม่พบคอลัมน์ 'review_text' ในไฟล์ CSV กรุณาตรวจสอบชื่อคอลัมน์")