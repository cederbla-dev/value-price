import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(page_title="미국주식 통합 분석 시스템 V4", layout="wide")

# --- 공통 데이터 수집 함수 (초이스스탁) ---
@st.cache_data(ttl=3600)
def fetch_choicestock(ticker):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(res.text))
        eps_data, per_data = pd.DataFrame(), pd.DataFrame()
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                eps_data = df.set_index(df.columns[0]).filter(like='EPS', axis=0).transpose()
            if df.iloc[:, 0].astype(str).str.contains('PER').any():
                per_data = df.set_index(df.columns[0]).filter(like='PER', axis=0).transpose()
        
        def adjust_date(dt_str):
            dt = pd.to_datetime(dt_str, format='%y.%m.%d', errors='coerce')
            if pd.isna(dt): return None
            return (dt.replace(day=1) - timedelta(days=1)).strftime('%Y-%m') if dt.day <= 5 else dt.strftime('%Y-%m')
        
        for df in [eps_data, per_data]:
            if not df.empty:
                df.index = [adjust_date(i) for i in df.index]
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        return eps_data, per_data
    except: return pd.DataFrame(), pd.DataFrame()

# --- 사이드바 설정 ---
st.sidebar.title("🛠️ 분석 설정")
menu = st.sidebar.radio("메뉴 선택", ["단일 종목 심층 분석", "종목 간 비교 (PE/EPS)", "섹터/ETF 비교"])

# --- [메뉴 1] 단일 종목 심층 분석 (파일 4, 5, 6, 10, 11번 로직) ---
if menu == "단일 종목 심층 분석":
    st.header("🔍 단일 종목 가치 평가")
    ticker = st.sidebar.text_input("티커 입력", value="AAPL").upper()
    base_year = st.sidebar.slider("지수화 기준 연도", 2017, 2024, 2018)
    
    eps_df, per_df = fetch_choicestock(ticker)
    if not eps_df.empty:
        price_raw = yf.download(ticker, start=f"{base_year-1}-01-01")['Close']
        price_m = price_raw.resample('M').last()
        price_m.index = price_m.index.strftime('%Y-%m')
        
        combined = pd.merge(eps_df.iloc[:,0], price_m, left_index=True, right_index=True)
        combined.columns = ['EPS', 'Price']
        
        # 파일 4, 6번의 핵심: 특정 시점 적정주가 계산
        st.subheader("1. 주가 vs 적정가 컨버전스")
        target_date = st.selectbox("가치 측정 기준점(Target Date) 선택", combined.index, index=len(combined)//2)
        
        b_eps = combined.loc[target_date, 'EPS']
        b_price = combined.loc[target_date, 'Price']
        mult = b_price / b_eps
        combined['Fair_Value'] = combined['EPS'] * mult
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined.index, y=combined['Price'], name="실제 주가", line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=combined.index, y=combined['Fair_Value'], name=f"적정가 (기준:{target_date})", line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
        
        # 괴리율 요약 (파일 10번)
        gap = ((combined['Price'].iloc[-1] / combined['Fair_Value'].iloc[-1]) - 1) * 100
        st.metric("현재 주가 괴리율", f"{gap:+.2f}%", delta="고평가" if gap > 0 else "저평가", delta_color="inverse")

# --- [메뉴 2] 종목 간 비교 (파일 9, 13번 로직) ---
elif menu == "종목 간 비교 (PE/EPS)":
    st.header("⚖️ 여러 종목 PE 및 EPS 성장률 비교")
    tickers = st.sidebar.text_input("비교할 티커들 (쉼표 구분)", value="AAPL, MSFT, GOOGL, NVDA").upper().replace(" ", "").split(",")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("종목별 PER 추이 비교")
        fig_pe = go.Figure()
        for t in tickers:
            _, per = fetch_choicestock(t)
            if not per.empty:
                fig_pe.add_trace(go.Scatter(x=per.index, y=per.iloc[:,0], name=t))
        st.plotly_chart(fig_pe, use_container_width=True)
        
    with col2:
        st.subheader("종목별 EPS 성장 추이 (지수화)")
        fig_eps = go.Figure()
        for t in tickers:
            eps, _ = fetch_choicestock(t)
            if not eps.empty:
                norm_eps = (eps.iloc[:,0] / eps.iloc[0,0]) * 100 # 첫 시점 100 기준
                fig_eps.add_trace(go.Scatter(x=eps.index, y=norm_eps, name=t))
        st.plotly_chart(fig_eps, use_container_width=True)

# --- [메뉴 3] 섹터/ETF 비교 (파일 12번 로직) ---
elif menu == "섹터/ETF 비교":
    st.header("Sector & Benchmark Performance")
    all_etfs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
    selected_etfs = st.multiselect("비교할 섹터/ETF 선택", all_etfs, default=["SPY", "QQQ", "XLK"])
    s_year = st.sidebar.number_input("시작 연도", value=2020)
    
    if selected_etfs:
        b_data = yf.download(selected_etfs, start=f"{s_year}-01-01")['Close']
        b_norm = (b_data / b_data.iloc[0]) * 100
        
        fig_sector = go.Figure()
        for col in b_norm.columns:
            width = 4 if col in ["SPY", "QQQ"] else 2
            fig_sector.add_trace(go.Scatter(x=b_norm.index, y=b_norm[col], name=col, line=dict(width=width)))
        
        fig_sector.update_layout(title=f"수익률 비교 (기준일: {b_norm.index[0].date()} = 100)")
        st.plotly_chart(fig_sector, use_container_width=True)
