import re
import pandas as pd
import streamlit as st
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.tag import pos_tag
from pythainlp.util import normalize


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Review Analyzer Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# EXECUTIVE DARK UI / LUXURY THEME CSS
# ============================================================

st.markdown("""
<style>
    /* Main Background & Base Typography */
    .stApp {
        background: linear-gradient(135deg, #0b0f17 0%, #111827 50%, #0f172a 100%);
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .main .block-container {
        max-width: 1380px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }

    p, span, label, .stMarkdown {
        color: #94a3b8;
    }

    /* Header Bar */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 24px;
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.35);
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin: 0;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-top: 2px;
    }

    .status-badge {
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.15);
    }

    /* Executive Cards */
    .card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
    }

    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .card-description {
        font-size: 13px;
        color: #64748b;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        min-height: 100px;
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #f8fafc;
        word-break: break-word;
    }

    .metric-helper {
        font-size: 11px;
        color: #475569;
        margin-top: 4px;
    }

    /* Section Styling */
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 28px 0 14px 0;
    }

    .section-title {
        font-size: 17px;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-caption {
        font-size: 12px;
        color: #64748b;
    }

    .result-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
    }

    .result-title {
        font-weight: 700;
        color: #cbd5e1;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Badges & Tags */
    .tag {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        margin: 4px 6px 4px 0;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    .tag-topic {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }

    .tag-positive {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }

    .tag-negative {
        background: rgba(244, 63, 94, 0.12);
        color: #fb7185;
        border: 1px solid rgba(251, 113, 133, 0.25);
    }

    .tag-neutral {
        background: rgba(148, 163, 184, 0.1);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* Streamlit Components Dark Overrides */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(15, 23, 42, 0.8) !important;
        color: #f8fafc !important;
        padding: 14px !important;
    }

    .stTextArea textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        min-height: 42px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: rgba(30, 41, 59, 0.8) !important;
        color: #f8fafc !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: rgba(51, 65, 85, 1) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 6px;
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(30, 41, 59, 1) !important;
        color: #38bdf8 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 16px;
    }

    /* Dataframe & Expanders */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }

    .stExpander {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
    }

    hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    .helper {
        color: #64748b;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED NLP RESOURCES
# ============================================================

@st.cache_data
def get_stopwords():
    """โหลด Stopwords ภาษาไทย และภาษาอังกฤษพื้นฐาน"""
    thai_sw = set(thai_stopwords())

    eng_sw = {
        "i", "me", "my", "myself", "we", "our", "ours", "you",
        "your", "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "what", "which", "who", "this",
        "that", "these", "those", "am", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "a", "an", "the", "and", "but",
        "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on",
        "off", "over", "under", "again", "further", "then", "once"
    }

    return thai_sw.union(eng_sw)


@st.cache_data
def get_dictionaries():
    """โหลด Dictionary สำหรับ Topic และ Sentiment (รองรับ ไทย/อังกฤษ)"""

    topic_keywords = {
        "รสชาติ/อาหาร": [
            "อร่อย", "แซ่บ", "กลมกล่อม", "จาง", "เค็ม", "หวาน",
            "เผ็ด", "จาน", "เมนู", "กุ้ง", "เนื้อ", "หมู", "ซุป",
            "delicious", "tasty", "yummy", "food", "dish", "menu",
            "soup", "meat", "pork", "shrimp", "sweet", "salty", "spicy"
        ],
        "การบริการ": [
            "พนักงาน", "บริการ", "ต้อนรับ", "เสิร์ฟ", "พูดจา",
            "รอนาน", "ช้า", "เร็ว", "ใส่ใจ",
            "service", "staff", "waiter", "waitress", "serve",
            "slow", "fast", "quick", "polite", "rude"
        ],
        "ราคา/ความคุ้มค่า": [
            "ราคา", "แพง", "ถูก", "คุ้ม", "ปริมาณ", "บิล", "บาท", "จานใหญ่",
            "price", "expensive", "cheap", "worth", "value", "cost",
            "bill", "portion"
        ],
        "บรรยากาศ/สถานที่": [
            "ร้าน", "ที่นั่ง", "แอร์", "ที่จอดรถ", "สะอาด", "สกปรก",
            "มุมถ่ายรูป", "วิว", "บรรยากาศ",
            "place", "shop", "restaurant", "atmosphere", "ambiance",
            "clean", "dirty", "seat", "parking", "view"
        ]
    }

    sentiment_keywords = {
        "คำชม (Positive)": [
            "อร่อย", "ดี", "ชอบ", "ประทับใจ", "เร็ว", "สะอาด", "คุ้ม",
            "น่ารัก", "สด", "ยอดเยี่ยม",
            "good", "great", "excellent", "love", "like", "awesome",
            "amazing", "fresh", "clean", "fast", "cheap"
        ],
        "คำติ (Negative)": [
            "ช้า", "แพง", "ไม่อร่อย", "สกปรก", "รอนาน", "แย่",
            "เค็มเกิน", "ห่วย", "จาง", "เหม็น",
            "bad", "terrible", "worst", "slow", "dirty", "expensive",
            "salty", "disappointed", "poor", "horrible"
        ]
    }

    return topic_keywords, sentiment_keywords


STOPWORDS = get_stopwords()
TOPIC_KEYWORDS, SENTIMENT_KEYWORDS = get_dictionaries()


# ============================================================
# NLP FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    """1. Regex & Cleansing: ลบ Noise, เบอร์โทร, URL, ปรับเป็นตัวพิมพ์เล็ก และลดคำลากเสียง"""

    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'(\d{2,4}[-\s]?\d{3,4}[-\s]?\d{3,4})', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = text.lower()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = normalize(text)

    return text.strip()


def process_tokens(text: str):
    """2. Tokenization & Normalization: ตัดคำและลบ Stopwords ทั้ง TH/EN"""

    raw_tokens = word_tokenize(
        text,
        engine="newmm",
        keep_whitespace=False
    )

    filtered_tokens = [
        w.strip()
        for w in raw_tokens
        if w.strip() not in STOPWORDS
        and len(w.strip()) > 1
    ]

    return raw_tokens, filtered_tokens


def extract_pos_and_entities(tokens: list):
    """3. POS & Entity Extraction"""

    tagged = pos_tag(
        tokens,
        engine="perceptron",
        corpus="orchid"
    )

    nouns = [
        word
        for word, tag in tagged
        if tag in ["NCMN", "NPRE", "NTNT"]
        or (word.isalnum() and word.isascii())
    ]

    adjectives = [
        word
        for word, tag in tagged
        if tag in ["VATT", "ADVN"]
    ]

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
            sentiments.append(
                f"{sentiment_type}: ({', '.join(matched)})"
            )

    return (
        detected_topics if detected_topics else ["ทั่วไป / General"],
        sentiments if sentiments else [
            "ไม่พบคำระบุความรู้สึกชัดเจน / Neutral"
        ]
    )


# ============================================================
# UI HELPERS
# ============================================================

def metric_card(label, value, helper=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-helper">{helper}</div>
    </div>
    """


def topic_tags(topics):
    html = ""
    for topic in topics:
        html += f'<span class="tag tag-topic">{topic}</span>'
    return html


def sentiment_tags(sentiments):
    html = ""
    for sentiment in sentiments:
        if "คำชม" in sentiment:
            css_class = "tag-positive"
        elif "คำติ" in sentiment:
            css_class = "tag-negative"
        else:
            css_class = "tag-neutral"

        html += f'<span class="tag {css_class}">{sentiment}</span>'
    return html


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="app-header">
    <div class="brand">
        <div class="brand-icon">⚡</div>
        <div>
            <div class="brand-title">Review Analyzer Pro</div>
            <div class="brand-subtitle">
                Executive NLP Intelligence System · Thai / English Multi-Topic Pipeline
            </div>
        </div>
    </div>

    <div class="status-badge">
        <span style="display:inline-block; width:8px; height:8px; background:#34d399; border-radius:50%;"></span>
        NLP Engine Active
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN TABS
# ============================================================

tab_single, tab_batch = st.tabs([
    "🔍  วิเคราะห์ข้อความเดี่ยว (Single Review)",
    "📂  วิเคราะห์ไฟล์ชุด (Batch CSV Processing)"
])


# ============================================================
# TAB 1: SINGLE TEXT PROCESSING
# ============================================================

with tab_single:

    st.markdown("""
    <div class="card">
        <div class="card-title">Single Review Analytics</div>
        <div class="card-description">
            ป้อนข้อความรีวิวเพื่อทำ Cleaning, Tokenization, POS Tagging, Topic Classification และ Sentiment Analysis แบบเรียลไทม์
        </div>
    </div>
    """, unsafe_allow_html=True)

    # State for quick sample inserts
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = (
            "The food was very delicious and fresh! "
            "But the service was extremely slow. "
            "Call 081-234-5678 https://example.com"
        )

    # Quick Sample Action Buttons (FEATURE ENHANCEMENT)
    col_sample1, col_sample2, _ = st.columns([1.5, 1.5, 5])
    with col_sample1:
        if st.button("📝 โหลดตัวอย่างภาษาไทย"):
            st.session_state["input_text"] = "อาหารอร่อยมาก กุ้งสดหวาน คุ้มราคา แต่พนักงานบริการช้านิดหน่อย ร้านสะอาดดีครับ"
            st.rerun()
    with col_sample2:
        if st.button("🌐 โหลดตัวอย่าง English"):
            st.session_state["input_text"] = "The pork soup was amazing and high quality! Highly recommended, cheap price."
            st.rerun()

    user_input = st.text_area(
        "ข้อความรีวิว",
        value=st.session_state["input_text"],
        height=130,
        label_visibility="collapsed",
        placeholder="พิมพ์ข้อความรีวิวที่นี่..."
    )

    col_button, col_hint = st.columns([1.2, 4])

    with col_button:
        analyze_button = st.button(
            "🚀 วิเคราะห์ข้อความ",
            type="primary",
            use_container_width=True
        )

    with col_hint:
        st.markdown(
            '<div class="helper" style="padding-top:12px;">'
            'รองรับการประมวลผลข้อความผสม TH/EN แบบอัตโนมัติ'
            '</div>',
            unsafe_allow_html=True
        )

    if analyze_button:

        if not user_input.strip():
            st.warning("กรุณากรอกข้อความก่อนเริ่มวิเคราะห์")

        else:
            # Pipeline Steps
            cleansed = clean_text(user_input)
            raw_tokens, filtered_tokens = process_tokens(cleansed)
            pos_tags, nouns, adjectives = extract_pos_and_entities(filtered_tokens)
            topics, sentiments = classify_topics_and_sentiment(filtered_tokens)

            st.markdown("""
            <div class="section-header">
                <div>
                    <div class="section-title">📊 สรุปผลการวิเคราะห์ (Analytics Overview)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.markdown(metric_card("Raw Tokens", len(raw_tokens), "จำนวนคำก่อนกรอง"), unsafe_allow_html=True)
            with m2:
                st.markdown(metric_card("Filtered Tokens", len(filtered_tokens), "หลังตัด Stopwords"), unsafe_allow_html=True)
            with m3:
                st.markdown(metric_card("Topics Detected", len(topics), "หัวข้อที่พบ"), unsafe_allow_html=True)
            with m4:
                st.markdown(metric_card("Sentiment Signals", len(sentiments), "สัญญาณความรู้สึก"), unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("""
                <div class="result-card">
                    <div class="result-title">🏷️ หัวข้อที่เกี่ยวข้อง (Topics)</div>
                    <div style="margin-top:14px;">
                """, unsafe_allow_html=True)
                st.markdown(topic_tags(topics), unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

            with c2:
                st.markdown("""
                <div class="result-card">
                    <div class="result-title">💬 ทัศนคติที่พบ (Sentiment)</div>
                    <div style="margin-top:14px;">
                """, unsafe_allow_html=True)
                st.markdown(sentiment_tags(sentiments), unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

            st.markdown("""
            <div class="section-header">
                <div class="section-title">⚙️ รายละเอียดเชิงลึก NLP (Detailed Execution)</div>
            </div>
            """, unsafe_allow_html=True)

            d1, d2 = st.columns(2)

            with d1:
                with st.expander("🧹 Text Cleansing Result", expanded=True):
                    st.code(cleansed, language=None)

                with st.expander("✂️ Tokenization Details"):
                    st.write("**คำทั้งหมด (Raw Tokens)**")
                    st.write(raw_tokens)
                    st.write("**คำหลังลบ Stopwords (Filtered)**")
                    st.write(filtered_tokens)

            with d2:
                with st.expander("📌 Key Extracted Elements (POS)", expanded=True):
                    st.write("**คำนาม / Entities (Nouns)**")
                    st.write(nouns if nouns else "-")
                    st.write("**คำคุณศัพท์ (Adjectives)**")
                    st.write(adjectives if adjectives else "-")

                with st.expander("🔤 Part-of-Speech Tagging Full Table"):
                    pos_df = pd.DataFrame(pos_tags, columns=["คำ", "POS Tag"])
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 2: BATCH CSV PROCESSING
# ============================================================

with tab_batch:

    st.markdown("""
    <div class="card">
        <div class="card-title">Batch Analytics & Reporting</div>
        <div class="card-description">
            อัปโหลดไฟล์ CSV ที่มีคอลัมน์ <b>review_text</b> เพื่อทำการวิเคราะห์เชิงปริมาณ รวบรวมสถิติ และส่งออกรายงาน
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample CSV Generator Helper (FEATURE ENHANCEMENT)
    col_upload, col_sample_download = st.columns([3, 1])

    with col_sample_download:
        sample_df = pd.DataFrame({
            "review_text": [
                "อาหารอร่อยมาก สดใหม่ แต่ราคาสูงไปนิด",
                "บริการแย่ ช้ามาก พนักงานพูดจาไม่ดี",
                "ร้านสวย สะอาด บรรยากาศดี มีที่จอดรถสะดวก",
                "Great food and cheap price! Highly recommended."
            ]
        })
        sample_csv = sample_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 โหลด CSV ตัวอย่าง",
            sample_csv,
            "sample_reviews.csv",
            "text/csv",
            use_container_width=True
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            "อัปโหลด CSV",
            type=["csv"],
            label_visibility="collapsed"
        )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"ไม่สามารถอ่านไฟล์ CSV ได้: {e}")
            df = None

        if df is not None:
            if df.empty:
                st.warning("ไฟล์ CSV ไม่มีข้อมูล")
            elif "review_text" not in df.columns:
                st.error("ไม่พบคอลัมน์ 'review_text' กรุณาตรวจสอบชื่อคอลัมน์ในไฟล์ CSV")
            else:
                m1, m2, m3 = st.columns(3)
                missing = df["review_text"].isna().sum()

                with m1:
                    st.markdown(metric_card("Total Reviews", f"{len(df):,}", "จำนวนรายการในไฟล์"), unsafe_allow_html=True)
                with m2:
                    st.markdown(metric_card("Columns", len(df.columns), "จำนวนคอลัมน์"), unsafe_allow_html=True)
                with m3:
                    st.markdown(metric_card("Missing Values", missing, "รายการที่ไม่มีข้อความ"), unsafe_allow_html=True)

                with st.expander("👀 ดูตัวอย่างข้อมูล (Preview Dataset)", expanded=False):
                    st.dataframe(df.head(5), use_container_width=True, hide_index=True)

                process_batch = st.button(
                    "⚡ ประมวลผลทั้งไฟล์ (Start Batch Processing)",
                    type="primary",
                    use_container_width=True
                )

                if process_batch:
                    progress = st.progress(0, text="กำลังวิเคราะห์ข้อมูล...")
                    results = []
                    total = len(df)

                    for index, text in enumerate(df["review_text"]):
                        original_text = "" if pd.isna(text) else str(text)
                        cleaned = clean_text(original_text)
                        _, filtered = process_tokens(cleaned)
                        topics, sentiments = classify_topics_and_sentiment(filtered)

                        results.append({
                            "ข้อความเดิม": original_text,
                            "ข้อความหลังคลีน": cleaned,
                            "หัวข้อ": ", ".join(topics),
                            "ข้อสังเกต Sentiment": " | ".join(sentiments)
                        })

                        progress.progress((index + 1) / total, text=f"กำลังวิเคราะห์ {index + 1:,} / {total:,}")

                    progress.empty()
                    result_df = pd.DataFrame(results)
                    st.success(f"วิเคราะห์ข้อมูลเสร็จสิ้นเรียบร้อยแล้ว {len(result_df):,} รายการ")

                    # Dashboard Summary
                    st.markdown("""
                    <div class="section-header">
                        <div class="section-title">📈 Executive Dashboard</div>
                    </div>
                    """, unsafe_allow_html=True)

                    all_topics = [
                        t.strip()
                        for sublist in result_df["หัวข้อ"].str.split(",")
                        for t in sublist if t.strip()
                    ]
                    topic_counts = pd.Series(all_topics).value_counts()
                    top_topic = topic_counts.index[0] if len(topic_counts) > 0 else "-"

                    pos_count = result_df["ข้อสังเกต Sentiment"].str.contains("คำชม", na=False).sum()
                    neg_count = result_df["ข้อสังเกต Sentiment"].str.contains("คำติ", na=False).sum()
                    neutral_count = max(len(result_df) - pos_count - neg_count, 0)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(metric_card("Processed Reviews", f"{len(result_df):,}", "รายการทั้งหมด"), unsafe_allow_html=True)
                    with m2:
                        st.markdown(metric_card("Top Topic", top_topic, "หัวข้อที่ถูกพูดถึงมากที่สุด"), unsafe_allow_html=True)
                    with m3:
                        st.markdown(metric_card("Positive Signals", f"{pos_count:,}", "จำนวนคำชมที่พบ"), unsafe_allow_html=True)
                    with m4:
                        st.markdown(metric_card("Negative Signals", f"{neg_count:,}", "จำนวนคำติที่พบ"), unsafe_allow_html=True)

                    # Charts
                    chart1, chart2 = st.columns(2)
                    with chart1:
                        st.markdown('<div class="card"><div class="card-title">📊 สัดส่วนหัวข้อที่ถูกพูดถึง</div>', unsafe_allow_html=True)
                        if not topic_counts.empty:
                            st.bar_chart(topic_counts)
                        else:
                            st.info("ไม่พบข้อมูล Topic")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with chart2:
                        st.markdown('<div class="card"><div class="card-title">💬 Sentiment Overview</div>', unsafe_allow_html=True)
                        sentiment_summary = pd.DataFrame(
                            {"จำนวน": [pos_count, neg_count, neutral_count]},
                            index=["Positive", "Negative", "Neutral"]
                        )
                        st.bar_chart(sentiment_summary)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Result Data Table
                    st.markdown("""
                    <div class="section-header">
                        <div class="section-title">📋 ผลการวิเคราะห์รายรายการ</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(result_df, use_container_width=True, hide_index=True, height=400)

                    # Export Button
                    csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 ดาวน์โหลดรายงานผลการวิเคราะห์ (Download Analytical CSV)",
                        csv_data,
                        "executive_nlp_analysis.csv",
                        "text/csv",
                        use_container_width=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="app-header">
<div class="brand">
<div class="brand-icon">⚡</div>
<div>
<div class="brand-title">Review Analyzer Pro</div>
<div class="brand-subtitle">
Executive NLP Intelligence System · Thai / English Multi-Topic Pipeline
</div>
</div>
</div>
<div class="status-badge">
<span style="display:inline-block; width:8px; height:8px; background:#34d399; border-radius:50%;"></span>
NLP Engine Active
</div>
</div>
""", unsafe_allow_html=True)