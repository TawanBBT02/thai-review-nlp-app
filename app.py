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
    page_title="Review Analyzer",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MODERN LIGHT UI
# ============================================================

st.markdown("""
<style>
    .stApp {
        background: #f7f8fa;
        color: #1f2937;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    h1, h2, h3 {
        color: #111827 !important;
        letter-spacing: -0.02em;
    }

    p, label, .stMarkdown {
        color: #4b5563;
    }

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    }

    .brand-title {
        font-size: 25px;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }

    .brand-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin-top: 2px;
    }

    .status-badge {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.035);
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
    }

    .card-description {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 16px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.025);
    }

    .metric-label {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 23px;
        font-weight: 700;
        color: #111827;
        word-break: break-word;
    }

    .metric-helper {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 4px;
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 24px 0 12px 0;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
    }

    .section-caption {
        font-size: 12px;
        color: #9ca3af;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        height: 100%;
    }

    .result-title {
        font-weight: 700;
        color: #111827;
        font-size: 15px;
    }

    .tag {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        margin: 3px 4px 3px 0;
        font-size: 12px;
        font-weight: 600;
    }

    .tag-topic {
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #dbeafe;
    }

    .tag-positive {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #d1fae5;
    }

    .tag-negative {
        background: #fff1f2;
        color: #be123c;
        border: 1px solid #ffe4e6;
    }

    .tag-neutral {
        background: #f3f4f6;
        color: #6b7280;
        border: 1px solid #e5e7eb;
    }

    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #d1d5db !important;
        background: #ffffff !important;
        color: #111827 !important;
        padding: 14px !important;
    }

    .stTextArea textarea:focus {
        border-color: #93c5fd !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.10) !important;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
        border: 1px solid #d1d5db;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 5px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #6b7280;
        font-size: 13px;
    }

    .stTabs [aria-selected="true"] {
        background: #f3f4f6;
        color: #111827 !important;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #d1d5db;
        border-radius: 14px;
        padding: 10px;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: #e5e7eb !important;
    }

    .helper {
        color: #9ca3af;
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

    # ลบ URL และเบอร์โทรศัพท์
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'(\d{2,4}[-\s]?\d{3,4}[-\s]?\d{3,4})', '', text)
    text = re.sub(r'[@#]\w+', '', text)

    # แปลงตัวอักษรภาษาอังกฤษเป็นตัวพิมพ์เล็ก
    text = text.lower()

    # ลดคำลากเสียง
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Normalize สระภาษาไทย
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
        <div class="brand-icon">🍔</div>
        <div>
            <div class="brand-title">Review Analyzer</div>
            <div class="brand-subtitle">
                ระบบคัดกรองและวิเคราะห์ข้อความรีวิวอาหารและสินค้า · Thai / English NLP
            </div>
        </div>
    </div>

    <div class="status-badge">
        ● NLP Engine Ready
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN TABS
# ============================================================

tab_single, tab_batch = st.tabs([
    "🔍  วิเคราะห์ข้อความเดี่ยว",
    "📂  วิเคราะห์ไฟล์ CSV"
])


# ============================================================
# TAB 1: SINGLE TEXT PROCESSING
# ============================================================

with tab_single:

    st.markdown("""
    <div class="card">
        <div class="card-title">
            วิเคราะห์รีวิวแบบข้อความเดียว
        </div>
        <div class="card-description">
            ป้อนข้อความรีวิว แล้วระบบจะทำความสะอาดข้อความ ตัดคำ
            วิเคราะห์หัวข้อ ความรู้สึก และ Key Elements
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area(
        "ข้อความรีวิว",
        value=(
            "The food was very delicious and fresh! "
            "But the service was extremely slow. "
            "Call 081-234-5678 https://example.com"
        ),
        height=140,
        label_visibility="collapsed",
        placeholder="พิมพ์ข้อความรีวิวที่นี่..."
    )

    col_button, col_hint = st.columns([1, 4])

    with col_button:
        analyze_button = st.button(
            "✨ วิเคราะห์ข้อความ",
            type="primary",
            use_container_width=True
        )

    with col_hint:
        st.markdown(
            '<div class="helper" style="padding-top:12px;">'
            'รองรับภาษาไทยและภาษาอังกฤษ'
            '</div>',
            unsafe_allow_html=True
        )

    if analyze_button:

        if not user_input.strip():
            st.warning("กรุณากรอกข้อความก่อนเริ่มวิเคราะห์")

        else:

            # 1. Cleansing
            cleansed = clean_text(user_input)

            # 2. Tokenization
            raw_tokens, filtered_tokens = process_tokens(cleansed)

            # 3. POS / Entities
            pos_tags, nouns, adjectives = extract_pos_and_entities(
                filtered_tokens
            )

            # 4. Topic / Sentiment
            topics, sentiments = classify_topics_and_sentiment(
                filtered_tokens
            )

            st.markdown("""
            <div class="section-header">
                <div>
                    <div class="section-title">ผลการวิเคราะห์</div>
                    <div class="section-caption">Analysis Summary</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.markdown(
                    metric_card(
                        "จำนวนคำก่อนกรอง",
                        len(raw_tokens),
                        "Raw tokens"
                    ),
                    unsafe_allow_html=True
                )

            with m2:
                st.markdown(
                    metric_card(
                        "จำนวนคำหลังกรอง",
                        len(filtered_tokens),
                        "After stopwords"
                    ),
                    unsafe_allow_html=True
                )

            with m3:
                st.markdown(
                    metric_card(
                        "หัวข้อที่พบ",
                        len(topics),
                        "Detected topics"
                    ),
                    unsafe_allow_html=True
                )

            with m4:
                st.markdown(
                    metric_card(
                        "Sentiment ที่พบ",
                        len(sentiments),
                        "Sentiment signals"
                    ),
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="section-title" style="margin-top:25px;">'
                'ภาพรวมความคิดเห็น'
                '</div>',
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("""
                <div class="result-card">
                    <div class="result-title">🏷️ หัวข้อที่เกี่ยวข้อง</div>
                    <div style="margin-top:12px;">
                """, unsafe_allow_html=True)

                st.markdown(
                    topic_tags(topics),
                    unsafe_allow_html=True
                )

                st.markdown("</div></div>", unsafe_allow_html=True)

            with c2:
                st.markdown("""
                <div class="result-card">
                    <div class="result-title">💬 Sentiment</div>
                    <div style="margin-top:12px;">
                """, unsafe_allow_html=True)

                st.markdown(
                    sentiment_tags(sentiments),
                    unsafe_allow_html=True
                )

                st.markdown("</div></div>", unsafe_allow_html=True)

            st.markdown(
                '<div class="section-title" style="margin-top:25px;">'
                'รายละเอียดการประมวลผล'
                '</div>',
                unsafe_allow_html=True
            )

            d1, d2 = st.columns(2)

            with d1:

                with st.expander(
                    "🧹 Cleansing — ข้อความหลังทำความสะอาด",
                    expanded=True
                ):
                    st.code(cleansed, language=None)

                with st.expander("✂️ Tokenization — ผลการตัดคำ"):

                    st.write("**คำทั้งหมด**")
                    st.write(raw_tokens)

                    st.write("**หลังลบ Stopwords**")
                    st.write(filtered_tokens)

            with d2:

                with st.expander(
                    "📌 Key Elements — POS Tagging",
                    expanded=True
                ):

                    st.write("**Nouns / Entities**")
                    st.write(nouns if nouns else "-")

                    st.write("**Adjectives**")
                    st.write(adjectives if adjectives else "-")

                with st.expander("🔤 POS Tags — รายละเอียด"):

                    pos_df = pd.DataFrame(
                        pos_tags,
                        columns=["คำ", "POS Tag"]
                    )

                    st.dataframe(
                        pos_df,
                        use_container_width=True,
                        hide_index=True
                    )


# ============================================================
# TAB 2: BATCH CSV PROCESSING
# ============================================================

with tab_batch:

    st.markdown("""
    <div class="card">
        <div class="card-title">
            วิเคราะห์รีวิวจำนวนมาก
        </div>
        <div class="card-description">
            อัปโหลดไฟล์ CSV ที่มีคอลัมน์ <b>review_text</b>
            เพื่อวิเคราะห์หลายรายการพร้อมกัน
        </div>
    </div>
    """, unsafe_allow_html=True)

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

                st.error(
                    "ไม่พบคอลัมน์ 'review_text' "
                    "กรุณาตรวจสอบชื่อคอลัมน์"
                )

            else:

                st.markdown(
                    '<div class="section-title" '
                    'style="margin-top:22px;">'
                    'ไฟล์ที่อัปโหลด'
                    '</div>',
                    unsafe_allow_html=True
                )

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.markdown(
                        metric_card(
                            "จำนวนรายการ",
                            f"{len(df):,}",
                            "Reviews"
                        ),
                        unsafe_allow_html=True
                    )

                with m2:
                    st.markdown(
                        metric_card(
                            "จำนวนคอลัมน์",
                            len(df.columns),
                            "Columns"
                        ),
                        unsafe_allow_html=True
                    )

                with m3:
                    missing = df["review_text"].isna().sum()

                    st.markdown(
                        metric_card(
                            "ข้อมูลว่าง",
                            missing,
                            "Missing review text"
                        ),
                        unsafe_allow_html=True
                    )

                with st.expander(
                    "👀 ดูตัวอย่างข้อมูล",
                    expanded=True
                ):
                    st.dataframe(
                        df.head(5),
                        use_container_width=True,
                        hide_index=True
                    )

                process_batch = st.button(
                    "⚡ ประมวลผลทั้งไฟล์",
                    type="primary",
                    use_container_width=True
                )

                if process_batch:

                    progress = st.progress(
                        0,
                        text="กำลังวิเคราะห์ข้อมูล..."
                    )

                    results = []
                    total = len(df)

                    for index, text in enumerate(df["review_text"]):

                        # รองรับ cell ว่างให้สะอาดกว่าเดิม
                        original_text = "" if pd.isna(text) else str(text)

                        cleaned = clean_text(original_text)

                        _, filtered = process_tokens(cleaned)

                        topics, sentiments = (
                            classify_topics_and_sentiment(filtered)
                        )

                        results.append({
                            "ข้อความเดิม": original_text,
                            "ข้อความหลังคลีน": cleaned,
                            "หัวข้อ": ", ".join(topics),
                            "ข้อสังเกต Sentiment": " | ".join(sentiments)
                        })

                        progress.progress(
                            (index + 1) / total,
                            text=f"กำลังวิเคราะห์ {index + 1:,} / {total:,}"
                        )

                    progress.empty()

                    result_df = pd.DataFrame(results)

                    st.success(
                        f"วิเคราะห์ข้อมูลเสร็จแล้ว {len(result_df):,} รายการ"
                    )

                    # ------------------------------------------------
                    # DASHBOARD
                    # ------------------------------------------------

                    st.markdown("""
                    <div class="section-header">
                        <div>
                            <div class="section-title">
                                📊 Analysis Dashboard
                            </div>
                            <div class="section-caption">
                                ภาพรวมผลการวิเคราะห์ทั้งหมด
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    all_topics = [
                        t.strip()
                        for sublist in result_df["หัวข้อ"].str.split(",")
                        for t in sublist
                        if t.strip()
                    ]

                    topic_counts = pd.Series(all_topics).value_counts()

                    top_topic = (
                        topic_counts.index[0]
                        if len(topic_counts) > 0
                        else "-"
                    )

                    pos_count = (
                        result_df["ข้อสังเกต Sentiment"]
                        .str.contains("คำชม", na=False)
                        .sum()
                    )

                    neg_count = (
                        result_df["ข้อสังเกต Sentiment"]
                        .str.contains("คำติ", na=False)
                        .sum()
                    )

                    # Neutral ที่ไม่พบ positive/negative signal
                    neutral_count = max(
                        len(result_df) - pos_count - neg_count,
                        0
                    )

                    m1, m2, m3, m4 = st.columns(4)

                    with m1:
                        st.markdown(
                            metric_card(
                                "รีวิวทั้งหมด",
                                f"{len(result_df):,}",
                                "Total reviews"
                            ),
                            unsafe_allow_html=True
                        )

                    with m2:
                        st.markdown(
                            metric_card(
                                "หัวข้อหลัก",
                                top_topic,
                                "Most detected topic"
                            ),
                            unsafe_allow_html=True
                        )

                    with m3:
                        st.markdown(
                            metric_card(
                                "คำชม",
                                f"{pos_count:,}",
                                "Positive signals"
                            ),
                            unsafe_allow_html=True
                        )

                    with m4:
                        st.markdown(
                            metric_card(
                                "คำติ",
                                f"{neg_count:,}",
                                "Negative signals"
                            ),
                            unsafe_allow_html=True
                        )

                    # ------------------------------------------------
                    # CHARTS
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title" '
                        'style="margin-top:25px;">'
                        'แนวโน้มจากข้อมูล'
                        '</div>',
                        unsafe_allow_html=True
                    )

                    chart1, chart2 = st.columns(2)

                    with chart1:

                        st.markdown("""
                        <div class="card">
                            <div class="card-title">
                                📈 รีวิวตามหัวข้อ
                            </div>
                            <div class="card-description">
                                จำนวนครั้งที่แต่ละ Topic ถูกตรวจพบ
                            </div>
                        """, unsafe_allow_html=True)

                        if not topic_counts.empty:
                            st.bar_chart(topic_counts)
                        else:
                            st.info("ยังไม่มีข้อมูล Topic")

                        st.markdown("</div>", unsafe_allow_html=True)

                    with chart2:

                        st.markdown("""
                        <div class="card">
                            <div class="card-title">
                                💬 Sentiment Overview
                            </div>
                            <div class="card-description">
                                เปรียบเทียบคำชม คำติ และ Neutral
                            </div>
                        """, unsafe_allow_html=True)

                        sentiment_summary = pd.DataFrame(
                            {
                                "จำนวน": [
                                    pos_count,
                                    neg_count,
                                    neutral_count
                                ]
                            },
                            index=[
                                "Positive",
                                "Negative",
                                "Neutral"
                            ]
                        )

                        st.bar_chart(sentiment_summary)

                        st.markdown("</div>", unsafe_allow_html=True)

                    # ------------------------------------------------
                    # RESULT TABLE
                    # ------------------------------------------------

                    st.markdown("""
                    <div class="section-header">
                        <div>
                            <div class="section-title">
                                📋 ผลการวิเคราะห์รายรายการ
                            </div>
                            <div class="section-caption">
                                ตรวจสอบรายละเอียดของแต่ละรีวิว
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                        height=450
                    )

                    # ------------------------------------------------
                    # EXPORT
                    # ------------------------------------------------

                    csv_data = (
                        result_df
                        .to_csv(index=False)
                        .encode("utf-8-sig")
                    )

                    st.download_button(
                        "📥 ดาวน์โหลดผลการวิเคราะห์ CSV",
                        csv_data,
                        "nlp_analysis_result.csv",
                        "text/csv",
                        use_container_width=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="
    text-align:center;
    color:#9ca3af;
    font-size:11px;
    margin-top:35px;
">
    Review Analyzer · Thai / English NLP
</div>
""", unsafe_allow_html=True)
