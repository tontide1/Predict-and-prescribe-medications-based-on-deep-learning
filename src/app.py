import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyvi import ViTokenizer
import numpy as np
import pandas as pd
import io

# Cấu hình CSS tùy chỉnh cho giao diện
custom_css = """
<style>
    body {
        background-color: #0e0e0e;
        color: #FFFFFF;
    }
    .title {
        color: #2ee65f;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 1.2em;
        text-align: center;
        color: #ffffff;
        margin-bottom: 30px;
    }
    textarea, .stTextArea textarea {
        background-color: #333333;
        color: #FFFFFF;
        border: 1px solid #555555;
    }
    .predict-button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
    }
    .footer a {
        color: #4CAF50;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .history-table {
        background-color: #1e1e1e;
        color: #FFFFFF;
        padding: 10px;
        border-radius: 8px;
    }
</style> 
"""

# Áp dụng CSS tùy chỉnh
st.markdown(custom_css, unsafe_allow_html=True)

# Tải mô hình và các bộ mã hóa với cache_resource
@st.cache_resource
def load_resources():
    model = load_model("C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data_test\\0.94_0.97\\trained_model.keras")
    vocab = joblib.load("C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data_test\\0.94_0.97\\vocab.pkl")
    label_encoder_disease = joblib.load("C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data_test\\0.94_0.97\\label_encoder_disease.pkl")
    label_encoder_prescription = joblib.load("C:\\Users\\tontide1\\my-data\\AI\\final\\data\\tokenizer_data_test\\0.94_0.97\\label_encoder_prescription.pkl")
    return model, vocab, label_encoder_disease, label_encoder_prescription

model, vocab, label_encoder_disease, label_encoder_prescription = load_resources()

def make_prediction(input_text, model, vocab, label_encoder_disease, label_encoder_prescription, max_len=100):
    if not input_text:
        return None, None
    
    tokenized_text = ViTokenizer.tokenize(input_text)
    sequence = []
    for word in tokenized_text.split():
        sequence.append(vocab.get(word, 0))  
    padded_sequence = pad_sequences([sequence], maxlen=max_len)
    
    predictions = model.predict(padded_sequence)
    disease_pred = np.argmax(predictions[0], axis=1)
    prescription_pred = np.argmax(predictions[1], axis=1)
    
    disease = label_encoder_disease.inverse_transform(disease_pred)[0]
    prescription = label_encoder_prescription.inverse_transform(prescription_pred)[0]
    
    return disease, prescription

# Khởi tạo lịch sử dự đoán trong session_state
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit App Layout
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="title">🩺 Dự Đoán Bệnh và Đề Xuất Thuốc</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Mô tả triệu chứng của bệnh nhân để nhận dự đoán bệnh và đề xuất thuốc.</div>', unsafe_allow_html=True)

# Khu vực nhập liệu và nút dự đoán
# Loại bỏ bố cục cột và sắp xếp theo chiều dọc
# Text area cho đầu vào
input_text = st.text_area("Nhập triệu chứng của bệnh nhân:", height=100)

if st.button("🔍 Dự Đoán", key="predict_button", help="Nhấn để dự đoán dựa trên triệu chứng đã nhập"):
    if input_text.strip():
        disease, prescription = make_prediction(input_text, model, vocab, label_encoder_disease, label_encoder_prescription)
        if disease and prescription:
            st.success(f"### 🧾 Bệnh Dự Đoán: **{disease}**")
            st.success(f"### 💊 Thuốc Đề Xuất: **{prescription}**")
            
            # Lưu lịch sử dự đoán
            st.session_state.history.append({
                'Triệu chứng nhập vào': input_text,
                'Bệnh dự đoán': disease,
                'Thuốc đề xuất': prescription
            })
        else:
            st.error("Dự đoán thất bại. Vui lòng kiểm tra lại đầu vào.")
    else:
        st.warning("Vui lòng nhập mô tả triệu chứng để thực hiện dự đoán.")

# Hiển thị lịch sử dự đoán bên dưới
st.markdown("### 📜 Lịch Sử Dự Đoán")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.table(history_df)
    
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        history_df.to_excel(writer, index=False, sheet_name='LichSuDuDoan')
    
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Tải Xuống Lịch Sử Dự Đoán",
        data=excel_buffer,
        file_name='lich_su_du_doan.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
else:
    st.info("Chưa có lịch sử dự đoán nào.")

st.markdown('</div>', unsafe_allow_html=True)

# Thông tin Sidebar
st.sidebar.title("📋 Giới Thiệu Website")
st.sidebar.markdown(
    """
    <div style="text-align: justify;">
        Website này sử dụng mô hình học máy được huấn luyện để dự đoán bệnh và đề xuất thuốc dựa trên triệu chứng.<br>
        Nhập mô tả triệu chứng vào khu vực chính và nhấn<br>
        <strong>🔍 Dự Đoán</strong> để nhận kết quả.
        <br>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    ---
    <div style="text-align: justify;">
        <strong>👨‍⚕️ LƯU Ý</strong><br>
        Công cụ này dành cho mục đích giáo dục và chưa thể thay thế tư vấn y tế chuyên nghiệp.
    </div>
    """,
    unsafe_allow_html=True
)