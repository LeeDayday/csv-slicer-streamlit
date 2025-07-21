import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="CSV Slicer", layout="centered")
st.title("📂 CSV Slicer")

# 상태 저장용 변수
if "result_list" not in st.session_state:
    st.session_state.result_list = []

# CSV 입력
csv_text = st.text_area("CSV 데이터 붙여넣기 (탭 구분)", height=300)

# 컬럼명 입력 → 엔터로도 실행
def extract():
    try:
        df = pd.read_csv(io.StringIO(csv_text), sep="\t")
        col = st.session_state.column_name.strip()
        if col not in df.columns:
            st.session_state.result_list = [f"❌ 컬럼 '{col}'이(가) 존재하지 않습니다."]
        else:
            st.session_state.result_list = df[col].dropna().astype(str).tolist()
    except Exception as e:
        st.session_state.result_list = [f"⚠️ 오류 발생: {e}"]

st.text_input("🔤 추출할 컬럼명", key="column_name", on_change=extract)

# 수동 추출 버튼
if st.button("🔍 추출"):
    extract()

# 결과 출력
if st.session_state.result_list:
    result_str = str(st.session_state.result_list)

    st.markdown("### 📤 추출 결과")
    st.success(f"총 {len(st.session_state.result_list)}개의 항목이 추출되었습니다.")
    st.code(result_str, language="python")

