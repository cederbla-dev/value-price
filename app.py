import streamlit as st
import pandas as pd
import numpy as np

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(layout="wide")
st.title("적정주가 계산 – 선택 셀 평균 예제")

# -----------------------------
# 1. 예제 데이터 생성
# -----------------------------
np.random.seed(0)

df = pd.DataFrame({
    "연도": [2019, 2020, 2021, 2022, 2023],
    "EPS": np.random.uniform(3, 10, 5).round(2),
    "PER": np.random.uniform(10, 25, 5).round(2),
    "ROE": np.random.uniform(8, 20, 5).round(2)
})

st.subheader("① 데이터 테이블")

# -----------------------------
# 2. AG Grid 옵션 설정
# -----------------------------
gb = GridOptionsBuilder.from_dataframe(df)

gb.configure_default_column(
    resizable=True,
    sortable=True,
    filter=True
)

# 행 선택 활성화 (다중 선택 가능)
gb.configure_selection(
    selection_mode="multiple",
    use_checkbox=True
)

grid_options = gb.build()

# -----------------------------
# 3. AG Grid 표시
# -----------------------------
grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=True,
    theme="alpine",
    height=300
)

selected_rows = grid_response["selected_rows"]

# -----------------------------
# 4. 선택된 셀 평균 계산
# -----------------------------
st.subheader("② 선택된 값 평균 계산")

target_column = st.selectbox(
    "평균을 계산할 컬럼을 선택하세요",
    ["EPS", "PER", "ROE"]
)

if selected_rows:
    selected_df = pd.DataFrame(selected_rows)

    avg_value = selected_df[target_column].mean()

    st.success(
        f"선택된 {target_column} 평균값: **{avg_value:.2f}**"
    )

    st.write("선택된 행 데이터")
    st.dataframe(selected_df)

else:
    st.info("왼쪽 체크박스를 이용해 행을 선택하세요.")
