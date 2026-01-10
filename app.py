import streamlit as st
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt
import numpy as np
import requests
from datetime import datetime
import warnings

# 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="미국주식 통합 분석 시스템", layout="wide")
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# [Core] 데이터 수집 및 회계 주기 동기화 엔진
# ==========================================

def normalize_to_standard_quarter(dt):
    """서로 다른 분기 마감일을 표준 분기(3, 6, 9, 12월)로 조정"""
    month, year = dt.month, dt.year
    if month in [1, 2, 3]:   new_month = 3
    elif month in [4, 5, 6]: new_month = 6
    elif month in [7, 8, 9]: new_month = 9
    else:                    new_month = 12
    return pd.Timestamp(year=year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

@st.cache_data(ttl=3600)
def fetch_ticker_full_data(ticker, show_q1, show_q2):
    """제공해주신 로직을 바탕으로 TTM PER과 예측치를 계산하는 함수"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        
        target_df = None
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains('PER').any():
                target_df = df.set_index(df.columns[0])
                break
        
        if target_df is None: return None, None, {}

        # 데이터 추출 및 전처리
        per_raw = pd.to_numeric(target_df[target_df.index.str.contains('PER')].transpose().iloc[:, 0], errors='coerce')
        eps_raw = pd.to_numeric(target_df[target_df.index.str.contains('EPS')].transpose().iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
        
        combined = pd.DataFrame({'PER': per_raw, 'EPS': eps_raw}).dropna()
        combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
        combined = combined.sort_index()
        
        historical_eps = combined['EPS'].tolist()
        stock = yf.Ticker(ticker)
        
        # 예측치 계산 로직 (제공해주신 슬라이딩 TTM 방식)
        est_dict = {}
        if show_q1:
            # fast_info 대신 history 사용하여 현재가 획득
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            est = stock.earnings_estimate
            
            if est is not None and not est.empty:
                last_dt = combined.index[-1]
                # Q1 예측
                q1_dt = last_dt + pd.DateOffset(months=3)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1
                
                # Q2 예측
                if show_q2:
                    q2_dt = q1_dt + pd.DateOffset(months=3)
                    ttm_eps_q2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[q2_dt, 'PER'] = current_price / ttm_eps_q2

        # 날짜 동기화 및 중복 제거
        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        
        return combined['PER'], combined['EPS'], est_dict

    except Exception as e:
        st.error(f"{ticker} 분석 중 오류: {e}")
        return None, None, {}

# ==========================================
# [Module 1] 개별 종목 밸류에이션
# ==========================================

def run_single_valuation():
    st.header("💎 종목별 밸류에이션 (개별)")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: ticker = st.text_input("티커 입력", "AAPL").upper().strip()
    with col2: base_year = st.number_input("기준 연도", 2017, 2025, 2017)
    with col3: include_est = st.radio("예측치 포함", ["None", "Current Q", "Next Q"], horizontal=True)

    if not ticker: return
    
    q1 = include_est in ["Current Q", "Next Q"]
    q2 = include_est == "Next Q"
    
    per_series, eps_series, _ = fetch_ticker_full_data(ticker, q1, q2)
    
    if per_series is None:
        st.warning("데이터를 불러올 수 없습니다."); return

    tab1, tab2 = st.tabs(["📉 PER 추세 분석", "📋 데이터 요약"])
    
    with tab1:
        plot_df = per_series[per_series.index >= f"{base_year}-01-01"]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_df.index, plot_df, marker='o', label=f"{ticker} PER")
        ax.axhline(plot_df.mean(), color='red', ls='--', label='Mean')
        ax.set_title(f"{ticker} PER Band")
        ax.legend()
        st.pyplot(fig)
    
    with tab2:
        st.write(f"### {ticker} Raw Data (Synced)")
        st.dataframe(per_series.iloc[::-1])

# ==========================================
# [Module 2] 종목 비교 분석 (제공 로직 완벽 이식)
# ==========================================

def run_comparison():
    st.header("⚖️ 종목 간 지표 비교 (회계 주기 동기화)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_input("비교 티커 (쉼표 구분)", "AAPL, MSFT, AVGO, NVDA")
        t_list = [x.strip().upper() for x in tickers_input.split(',')]
    with col2:
        include_est_comp = st.radio("예측치 포함 (비교)", ["None", "Current Q", "Next Q"], horizontal=True)

    comp_mode = st.selectbox("비교 지표 선택", ["상대 PER 추세", "EPS 성장률 비교"])
    start_year = st.number_input("기준 연도 설정", 2010, 2025, 2017)

    if st.button("비교 분석 실행"):
        q1 = include_est_comp in ["Current Q", "Next Q"]
        q2 = include_est_comp == "Next Q"
        
        master_df = pd.DataFrame()
        
        for t in t_list:
            per_s, eps_s, _ = fetch_ticker_full_data(t, q1, q2)
            if per_s is not None:
                master_df[t] = per_s if comp_mode == "상대 PER 추세" else eps_s

        if master_df.empty:
            st.error("분석할 데이터가 없습니다."); return

        # 필터링 및 정규화
        master_df = master_df[master_df.index >= f"{start_year}-01-01"].sort_index()
        indexed_df = (master_df / master_df.iloc[0]) * 100
        
        # 차트 생성
        fig, ax = plt.subplots(figsize=(15, 8))
        x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_df.index]
        x_indices = np.arange(len(indexed_df))

        forecast_count = (1 if q1 else 0) + (1 if q2 else 0)

        for ticker in indexed_df.columns:
            series = indexed_df[ticker].dropna()
            valid_indices = [indexed_df.index.get_loc(dt) for dt in series.index]
            
            # 실제/예측 데이터 분리 시각화
            if forecast_count > 0:
                hist_idx = valid_indices[:-forecast_count]
                hist_val = series.values[:-forecast_count]
                pred_idx = valid_indices[-forecast_count-1:]
                pred_val = series.values[-forecast_count-1:]
                
                line, = ax.plot(hist_idx, hist_val, marker='o', label=f"{ticker} ({series.iloc[-1]:.1f})")
                ax.plot(pred_idx, pred_val, ls='--', color=line.get_color(), alpha=0.7)
                ax.scatter(valid_indices[-forecast_count:], series.values[-forecast_count:], marker='D', s=60, color=line.get_color())
            else:
                ax.plot(valid_indices, series.values, marker='o', label=f"{ticker} ({series.iloc[-1]:.1f})")

        ax.axhline(100, color='black', alpha=0.5, lw=1)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_title(f"{comp_mode} (Base 100 at {start_year})")
        ax.legend(loc='upper left')
        st.pyplot(fig)

# ==========================================
# [Module 3] 섹터 수익률
# ==========================================

def run_sector_perf():
    st.header("📊 섹터 및 지수 수익률")
    all_tickers = ["SPY", "QQQ", "XLK", "XLV", "XLY", "XLF", "XLI", "XLP", "XLE", "XLC", "XLB", "XLU", "XLRE"]
    selected = st.multiselect("비교 대상 선택", all_tickers, default=["SPY", "QQQ", "XLK"])
    start_date = st.date_input("비교 시작일", value=datetime(2017, 1, 1))
    
    if st.button("수익률 차트 생성"):
        combined_price = pd.DataFrame()
        for t in selected:
            s = yf.Ticker(t).history(start=start_date)['Close']
            if not s.empty: combined_price[t] = (s / s.iloc[0]) * 100
        if not combined_price.empty:
            st.line_chart(combined_price)

# ==========================================
# 메인 메뉴
# ==========================================

def main():
    st.sidebar.title("🇺🇸 주식 분석 터미널")
    menu = st.sidebar.selectbox("메뉴 선택", ["홈", "개별 종목 밸류에이션", "종목 비교 분석 (Sync)", "섹터/지수 수익률"])
    
    if menu == "홈":
        st.title("US Stock Analysis System")
        st.info("회계 주기가 다른 종목들을 표준 분기로 동기화하여 비교 분석합니다.")
    elif menu == "개별 종목 밸류에이션": run_single_valuation()
    elif menu == "종목 비교 분석 (Sync)": run_comparison()
    elif menu == "섹터/지수 수익률": run_sector_perf()

if __name__ == "__main__":
    main()
