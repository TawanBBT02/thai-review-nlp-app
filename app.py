import re
import pandas as pd
import streamlit as st
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.tag import pos_tag
from pythainlp.util import normalize

# --- CONFIGURATION ---
st.set_page_config(
    page_title="ระบบวิเคราะห์ข้อความรีวิวร้านอาหาร",
    page_icon="🍔",
    layout="wide"
)

STOPWORDS = set(thai_stopwords())

# พจนานุกรมสำหรับ Topic Identification & Sentiment
TOPIC_KEYWORDS = {
    "รสชาติ/อาหาร": ["อร่อย", "แซ่บ", "กลมกล่อม", "จาง", "เค็ม", "หวาน", "เผ็ด", "จาน", "เมนู", "กุ้ง", "เนื้อ", "หมู", "ซุป"],
    "การบริการ": ["พนักงาน", "บริการ", "ต้อนรับ", "เสิร์ฟ", "พูดจา", "รอนาน", "ช้า", "เร็ว", "ใส่ใจ"],
    "ราคา/ความคุ้มค่า": ["ราคา", "แพง", "ถูก", "คุ้ม", "ปริมาณ", "บิล", "บาท", "จานใหญ่"],
    "บรรยากาศ/สถานที่": ["ร้าน", "ที่นั่ง", "แอร์", "ที่จอดรถ", "สะอาด", "สกปรก", "มุมถ่ายรูป", "วิว", "บรรยากาศ"]
}

SENTIMENT_KEYWORDS = {
    "คำชม (Positive)": ["อร่อย", "ดี", "ชอบ", "ประทับใจ", "เร็ว", "สะอาด", "คุ้ม", "น่ารัก", "สด", "ยอดเยี่ยม"],
    "คำติ (Negative)": ["ช้า", "แพง", "ไม่อร่อย", "สกปรก", "รอนาน", "แย่", "เค็มเกิน", "ห่วย", "จาง", "เหม็น"]
}

# --- NLP FUNCTIONS ---
def clean_text(text: str) -> str:
    """1. Regex & Cleansing: ลบ Noise, เบอร์โทร, URL และลดคำลากเสียง"""
    if not isinstance(text, str):
        return ""
    # ลบ URL
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # ลบ เบอร์โทรศัพท์ (เลขไทยและอารบิก)
    text = re.sub(r'(\d{2,4}[-\s]?\d{3,4}[-\s]?\d{3,4})', '', text)
    # ลบ User Mention หรือ Hashtag
    text = re.sub(r'[@#]\w+', '', text)
    # ตัดคำลากเสียง เช่น "อร่อยยยยย" -> "อร่อย"
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Normalize อักขระภาษาไทย (แก้สระซ้อน)
    text = normalize(text)
    return text.strip()

def process_tokens(text: str):
    """2. Tokenization & Normalization: ตัดคำและลบ Stopwords"""
    raw_tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    filtered_tokens = [w for w in raw_tokens if w not in STOPWORDS and len(w.strip()) > 1]
    return raw_tokens, filtered_tokens

def extract_pos_and_entities(tokens: list):
    """3. POS & NER/Keyword Extraction: สกัดคำคุณศัพท์ (ADJ) และคำนาม (NOUN)"""
    tagged = pos_tag(tokens, engine='perceptron', corpus='orchid')
    nouns = [word for word, tag in tagged if tag in ['NCMN', 'NPRE', 'NTNT']]
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
        detected_topics if detected_topics else ["ทั่วไป"],
        sentiments if sentiments else ["ไม่พบคำระบุความรู้สึกชัดเจน"]
    )

# --- UI LAYOUT ---
st.title("🍔 ระบบคัดกรองและวิเคราะห์ข้อความรีวิวร้านอาหาร")
st.subheader("Text Processing & NLP Application")

tab1, tab2 = st.tabs(["🔍 วิเคราะห์ข้อความเดี่ยว", "📂 วิเคราะห์ไฟล์ CSV"])

# --- TAB 1: SINGLE TEXT PROCESSING ---
with tab1:
    user_input = st.text_area(
        "กรอกข้อความรีวิวที่ต้องการวิเคราะห์:",
        value="อาหารอร่อยยยยยมากกก กุ้งตัวใหญ่สดมาก แต่พนักงานบริการช้า รอนานเกิน 30 นาที โทรตามเบอร์ 081-234-5678 ก็ไม่มีคนรับ https://example.com",
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
                st.write("**หลังลบ Stopwords:**", filtered_tokens)

            with col2:
                st.markdown("### 🏷️ 3. การจำแนกหัวข้อและ Sentiment")
                st.write("**หัวข้อที่เกี่ยวข้อง (Topics):**")
                for t in topics:
                    st.success(f"- {t}")
                st.write("**คำชม / คำติ ที่พบ:**")
                for s in sentiments:
                    st.warning(f"- {s}")

                st.markdown("### 📌 4. Key Elements (POS Tagging)")
                st.write("**คำนาม/เมนู/สถานที่ (Nouns):**", nouns)
                st.write("**คำบรรยาย/คุณลักษณะ (Adjectives):**", adjectives)

# --- TAB 2: BATCH CSV PROCESSING ---
with tab2:
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV (ต้องมีคอลัมน์ชื่อ 'review_text')", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "review_text" in df.columns:
            st.write("ตัวอย่างข้อมูลที่อัปโหลด:", df.head(3))
            
            if st.button("ประมวลผลทั้งไฟล์"):
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
                st.dataframe(result_df, use_container_width=True)
                
                # ปุ่มดาวน์โหลดผลลัพธ์
                csv_data = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("ดาวน์โหลดผลการวิเคราะห์ (CSV)", csv_data, "nlp_analysis_result.csv", "text/csv")
        else:
            st.error("ไม่พบคอลัมน์ 'review_text' ในไฟล์ CSV")