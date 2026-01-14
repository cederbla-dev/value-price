# app-v6-Test1.py  (st-aggrid 제거 / 옵션1 구현 최종본)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import io
from datetime import timedelta

st.set_page_config(page_title="PER 기반 적정주가 분석", layout="wide")
st.title("📊 개별종목 적정주가 분석")

# -------------------------------------------------
# 공통 유틸
# -------------------------------------------------
def fetch_per_eps_from_choicestock(ticker):
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    dfs = pd.read_html(io.StringIO(r.text))

    per_df, eps_df = None, None
    for df in dfs:
        if df.iloc[:, 0].astype(str).str.contains("PER").any():
            per_df = df.set_index(df.columns[0]).transpose()
        if df.iloc[:, 0].astype(str).str.contains("EPS").any():
            eps_df = df.set_index(df.columns[0]).transpose()

    if per_df is None or eps_df is None:
        return None

    out = pd.DataFrame({
        "PER": pd.to_numeric(per_df.iloc[:, 0], errors="coerce"),
        "EPS": pd.to_numeric(
            eps_df.iloc[:, 0].astype(str).str.replace(",", ""),
            errors="coerce"
        )
    }).dropna()

    out.index = pd.to_datetime(out.index, format="%y.%m.%d", errors="coerce")
    out = out.dropna().sort_index()
    return out


def quarter_label(dt):
    m = dt.month if dt.day > 5 else (dt - timedelta(days=5)).month
    q = (m - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def recent_4q_eps_sum(ticker, mode):
    base = fetch_per_eps_from_choicestock(ticker)
    if base is None or base.empty:
        return None

    actual_eps = base["EPS"].iloc[::-1].tolist()

    stock = yf.Ticker(ticker)
    est = stock.earnings_estimate

    if mode == "None":
        return sum(actual_eps[:4])

    if mode == "현재 분기 예측":
        return sum(actual_eps[:3]) + est["avg"].iloc[0]

    if mode == "다음 분기 예측":
        return sum(actual_eps[:2]) + est["avg"].iloc[0] + est["avg"].iloc[1]


# -------------------------------------------------
# UI
# -------------------------------------------------
with st.container(border=True):
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        ticker = st.text_input("🏢 티커", "MSFT").upper()
    with c2:
        start_year = st.number_input("📅 기준 연도", 2010, 2025, 2017)
    with c3:
        predict_mode = st.radio(
            "🔮 미래 예측 옵션",
            ("None", "현재 분기 예측", "다음 분기 예측"),
            horizontal=True
        )

    run = st.button("데이터 분석 실행", type="primary", use_container_width=True)

# -------------------------------------------------
# 분석 실행
# -------------------------------------------------
if run and ticker:
    data = fetch_per_eps_from_choicestock(ticker)

    if data is None or data.empty:
        st.error("데이터를 불러오지 못했습니다.")
        st.stop()

    data = data[data.index.year >= start_year].copy()
    data["Quarter"] = data.index.map(quarter_label)

    # Pivot
    data["Year"] = data["Quarter"].str[:4]
    data["Q"] = data["Quarter"].str[-2:]

    pivot = data.pivot(index="Year", columns="Q", values="PER")
    pivot = pivot.reindex(columns=["Q1", "Q2", "Q3", "Q4"])

    st.subheader("📋 PER 테이블 (연도별 체크 선택)")
    st.caption("✔ 체크한 연도의 모든 분기 PER이 계산에 포함됩니다.")

    selected_pers = []

    for year in pivot.index:
        cols = st.columns([0.5, 1, 1, 1, 1, 1])
        checked = cols[0].checkbox("", key=f"chk_{year}")
        cols[1].markdown(f"**{year}**")

        for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            val = pivot.loc[year, q]
            cols[i + 2].write("-" if pd.isna(val) else f"{val:.1f}")

            if checked and pd.notna(val):
                selected_pers.append(val)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("② 선택 PER 평균 구하기"):
            if not selected_pers:
                st.warning("선택된 PER이 없습니다.")
            else:
                mean_per = np.mean(selected_pers)
                st.session_state["mean_per"] = mean_per
                st.success(f"평균 PER: **{mean_per:.2f}x**")

    with col_b:
        if st.button("③ 적정주가 구하기"):
            if "mean_per" not in st.session_state:
                st.warning("먼저 평균 PER을 계산하세요.")
            else:
                eps_sum = recent_4q_eps_sum(ticker, predict_mode)
                if eps_sum is None:
                    st.error("EPS 합 계산 실패")
                else:
                    fair_price = st.session_state["mean_per"] * eps_sum
                    st.success(
                        f"📌 적정주가 = {st.session_state['mean_per']:.2f} × "
                        f"{eps_sum:.2f} = **${fair_price:.2f}**"
                    )
                    st.caption(f"미래 예측 옵션: {predict_mode}")
