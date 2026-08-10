# ⚡ Review Analyzer Pro
### 🍔 Food & Product Review NLP Extractor (Executive Dark Edition)

ระบบ Web Application สำหรับคัดกรอง สกัดข้อมูล และวิเคราะห์ข้อความรีวิวอาหารและสินค้าแบบสองภาษา (ไทย/อังกฤษ) พัฒนาด้วย Python, PyThaiNLP และ Streamlit มาพร้อมกับอินเทอร์เฟซสไตล์ Executive Dark Mode ที่หรูหราและสบายตา

---

## 📌 คุณสมบัติของระบบ (NLP Pipeline & Features)

1. **Regex & Cleansing:** ลบ URL, เบอร์โทรศัพท์, แฮชแท็ก, สัญลักษณ์รบกวน และปรับคำลากเสียง (เช่น `อร่อยยยย` -> `อร่อย`) พร้อมปรับเป็นตัวพิมพ์เล็กสำหรับภาษาอังกฤษ
2. **Tokenization & Normalization:** ตัดคำภาษาไทย/อังกฤษ ด้วย PyThaiNLP (Engine `newmm`) จัดการสระซ้อน และลบคำหยุด (Stopwords)
3. **Topic Identification:** จัดกลุ่มหัวข้อรีวิวอัตโนมัติแบบ Multi-topic (รสชาติ/อาหาร, การบริการ, ราคา/ความคุ้มค่า, บรรยากาศ/สถานที่)
4. **POS & Entity Extraction:** ดึงคำนาม (เมนู/สถานที่) และคำบรรยาย (คำคุณศัพท์) ด้วย Part-of-Speech Tagging
5. **Batch Processing & Executive Dashboard:** รองรับการประมวลผลไฟล์ CSV พร้อมแสดงผลสรุป กราฟสถิติ และปุ่มส่งออกรายงาน (Export CSV)

---

## ⚙️ การทำงานของระบบ (NLP Pipeline Workflow)

```text
Input Text ──► 1. Cleansing ──► 2. Tokenization ──► 3. POS Extraction ──► 4. Topic & Sentiment ──► Dashboard Output