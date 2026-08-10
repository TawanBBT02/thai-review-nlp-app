# thai-review-nlp-app
# 🍔 Food & Product Review NLP Extractor

ระบบ Web Application สำหรับคัดกรองและวิเคราะห์ข้อความรีวิวอาหารและสินค้า พัฒนาด้วย Python, PyThaiNLP และ Streamlit

## 📌 คุณสมบัติของระบบ (NLP Pipeline)
1. **Regex & Cleansing**: ลบ URL, เบอร์โทรศัพท์, โน้ต/สัญลักษณ์รบกวน และปรับคำลากเสียง (เช่น `อร่อยยยย` -> `อร่อย`)
2. **Tokenization & Normalization**: ตัดคำภาษาไทย ลบ Stopwords และจัดการสระซ้อน
3. **Topic Identification**: จัดกลุ่มหัวข้อรีวิวอัตโนมัติ (รสชาติ/อาหาร, การบริการ, ราคา, บรรยากาศ)
4. **POS & Entity Extraction**: ดึงคำนาม (เมนู/สถานที่) และคำบรรยาย (คำคุณศัพท์)

---

## 🤖 AI Prompts ที่ใช้ในการพัฒนาระบบ
ตัวอย่าง Prompt ที่ใช้สั่งการ AI Assistant ในการช่วยเขียนโปรแกรม:

> **Prompt 1 (สร้าง Pipeline NLP):**
> "ช่วยเขียนฟังก์ชัน Python สำหรับทำ Text Cleansing ข้อความภาษาไทย โดยใช้ Regex ลบ URL, เบอร์โทรศัพท์ และลดคำลากเสียง เช่น 'อร่อยยย' ให้เหลือ 'อร่อย' พร้อมทั้งใช้ PyThaiNLP ตัดคำและลบ Stopwords"

> **Prompt 2 (การทำ UI ด้วย Streamlit):**
> "เขียนโค้ด Streamlit แสดงผลเป็น 2 Tabs โดย Tab แรกรับข้อความ Input เดี่ยวเพื่อแสดงผลการวิเคราะห์ NLP แต่ละ Step ส่วน Tab 2 สามารถ Upload ไฟล์ CSV แล้วรัน Batch Processing พร้อมปุ่มดาวน์โหลดผลลัพธ์ CSV"

---

## 🛠️ วิธีการติดตั้งและใช้งานบนเครื่องท้องถิ่น (Local Machine)

1. Clone Repository นี้:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd <REPOSITORY_FOLDER>