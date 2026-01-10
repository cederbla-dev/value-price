import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import io
import numpy as np
from datetime import datetime, timedelta
import warnings

# 1. 설정 및 한글 폰트(가능한 경우) 대응
warnings.filterwarnings("ignore")
st.set_page_config(page_title="미국주식 통합 분석 시스템", layout="wide")
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# [Core] 데이터 수집 및 전처리 엔진
# ==========================================

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    """주가, 과거 EPS/PER(ChoiceStock), 예측치(Yahoo) 통합 수집"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    
    try:
        # A. ChoiceStock 크롤링
        res = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(res.text))
        
        eps_raw = pd.DataFrame()
        per_raw = pd.DataFrame()
        
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                temp = df.set_index(df.columns[0]).filter(like='EPS', axis=0).transpose()
                temp.index = pd.to_datetime(temp.index, format='%y.%m.%d', errors='coerce')
                eps_raw = temp.dropna().sort_index()
                eps_raw.columns = ['EPS']
            if df.iloc[:, 0].astype(str).str.contains('PER').any():
                temp = df.set_index(df.columns[0]).filter(like='PER', axis=0).transpose()
                temp.index = pd.to_datetime(temp.index, format='%y.%m.%d', errors='coerce')
                per_raw = temp.dropna().sort_index()
                per_raw.columns = ['PER']
        
        # B. Yahoo 주가 및 예측치
        stock = yf.Ticker(ticker)
        price = stock.history(start="2016-10-01")['Close']
        if price.index.tz is not None: price.index = price.index.tz_localize(None)
        
        est = stock.earnings_estimate
        est_dict = {}
        if est is not None and not est.empty:
            est_dict = {
                'curr_q': est.loc['0q', 'avg'] if '0q' in est.index else None,
                'next_q': est.loc['+1q', 'avg'] if '+1q' in est.index else None,
                'curr_y': est.loc['0y', 'avg'] if '0y' in est.index else None
            }
            
        return eps_raw, per_raw, price, est_dict
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), {}

# ==========================================
# [Module 1] 개별 종목 정밀 분석 (File 6, 8, 10, 11 통합)
# ==========================================

def run_single_valuation():
    st.header("💎 종목별 밸류에이션 및 적정주가")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: ticker = st.text_input("티커 입력", "AAPL").upper().strip()
    with col2: base_year = st.number_input("기준 시작 연도", 2017, 2025, 2017)
    with col3: include_est = st.radio("예측치 포함", ["None", "Current Q", "Next Q"], horizontal=True)

    if not ticker: return

    eps_df, per_df, price_ser, ests = fetch_stock_data(ticker)
    if eps_df.empty: return

    # --- 데이터 병합 및 예측치 반영 ---
    eps_combined = eps_df[eps_df.index >= f"{base_year}-01-01"].copy()
    eps_combined['Type'] = 'Actual'
    
    if include_est != "None" and ests:
        last_dt = eps_combined.index[-1]
        if ests['curr_q']:
            eps_combined.loc[last_dt + pd.DateOffset(months=3)] = [ests['curr_q'], 'Estimate']
        if include_est == "Next Q" and ests['next_q']:
            eps_combined.loc[last_dt + pd.DateOffset(months=6)] = [ests['next_q'], 'Estimate']

    tab1, tab2, tab3 = st.tabs(["📊 적정주가 시뮬레이션", "📈 PER 밴드 분석", "📋 연간 요약 & PEG"])

    with tab1:
        st.subheader("연도별 시작점 기준 Fair Value 분석")
        
        # 주가 데이터 결합 (월말 기준)
        price_m = price_ser.resample('M').last()
        df_val = eps_combined.join(price_m, how='left')
        df_val['Close'] = df_val['Close'].ffill() # 주가 누락 방지
        
        summary_rows = []
        target_date = df_val.index[-1]
        current_price = price_ser.iloc[-1]

        # 각 연도별로 적정주가 계산 루프 (File 6 로직)
        for yr in range(base_year, datetime.now().year + 1):
            subset = df_val[df_val.index >= f"{yr}-01-01"]
            if len(subset) < 2 or subset.iloc[0]['EPS'] <= 0: continue
            
            base_eps = subset.iloc[0]['EPS']
            base_p = subset.iloc[0]['Close']
            mult = base_p / base_eps
            
            fair_val = subset['EPS'].iloc[-1] * mult
            gap = ((current_price - fair_val) / fair_val) * 100
            
            summary_rows.append({
                "기준연도": yr, "기준PER": f"{mult:.1f}x", 
                "Target EPS": f"${subset['EPS'].iloc[-1]:.2f}",
                "적정주가": round(fair_val, 2), "현재가": round(current_price, 2),
                "괴리율": f"{gap:+.2f}%", "판단": "고평가" if gap > 0 else "저평가"
            })

        st.table(pd.DataFrame(summary_rows))
        
        # 메인 그래프 (첫 번째 기준 연도 사용)
        if summary_rows:
            first_yr = summary_rows[0]['기준연도']
            mult = float(summary_rows[0]['기준PER'].replace('x',''))
            df_plot = df_val[df_val.index >= f"{first_yr}-01-01"].copy()
            df_plot['Fair'] = df_plot['EPS'] * mult
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_plot.index, df_plot['Close'], label='Market Price', marker='o', color='royalblue')
            ax.plot(df_plot.index, df_plot['Fair'], label=f'Fair Value ({mult:.1f}x)', linestyle='--', color='crimson')
            
            # 예측 영역 배경 표시
            est_idx = df_plot[df_plot['Type'] == 'Estimate'].index
            if not est_idx.empty:
                ax.axvspan(est_idx[0] - pd.DateOffset(days=15), est_idx[-1] + pd.DateOffset(days=15), color='orange', alpha=0.1, label='Estimates')
            
            ax.set_title(f"{ticker} Valuation Chart (Base: {first_yr})")
            ax.legend()
            st.pyplot(fig)

    with tab2:
        # File 8: PER Mean vs Median
        st.subheader("과거 PER 추이 및 통계")
        p_sub = per_df[per_df.index >= f"{base_year}-01-01"]
        if not p_sub.empty:
            avg_p = p_sub['PER'].mean()
            med_p = p_sub['PER'].median()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(p_sub.index, p_sub['PER'], marker='o', color='darkslategray', label='PER')
            ax.axhline(avg_p, color='red', linestyle='--', label=f'Mean: {avg_p:.2f}')
            ax.axhline(med_p, color='purple', linestyle='-.', label=f'Median: {med_p:.2f}')
            ax.legend()
            st.pyplot(fig)
            st.write(f"현재 PER: **{p_sub['PER'].iloc[-1]:.2f}** | 평균 대비: **{((p_sub['PER'].iloc[-1]/avg_p)-1)*100:+.1f}%**")

    with tab3:
        # File 10, 11: Annual Summary & PEG
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("### 4분기 합산(TTM) 분석")
            ttm_eps = eps_df['EPS'].rolling(4).sum().dropna()
            st.dataframe(ttm_eps.iloc[::-1].head(10))
        with col_b:
            st.write("### PEG 분석 (Price/Earnings to Growth)")
            if len(eps_df) >= 8:
                curr_ttm = eps_df['EPS'].iloc[-4:].sum()
                past_ttm = eps_df['EPS'].iloc[-8:-4].sum()
                growth = ((curr_ttm / past_ttm) - 1) * 100
                curr_per = price_ser.iloc[-1] / curr_ttm
                peg = curr_per / growth if growth > 0 else 0
                st.metric("Growth (YoY)", f"{growth:.1f}%")
                st.metric("PEG Ratio", f"{peg:.2f}")

# ==========================================
# [Module 2] 비교 분석 (File 9, 12, 13 통합)
# ==========================================

def run_comparison():
    st.header("⚖️ 종목 간 비교 분석")
    
    comp_mode = st.radio("비교 모드", ["EPS 성장률 비교", "섹터/지수 수익률", "상대 PER 추세"], horizontal=True)
    
    tickers_input = st.text_input("비교 티커 (쉼표로 구분)", "AAPL, MSFT, GOOGL, NVDA")
    t_list = [x.strip().upper() for x in tickers_input.split(',')]
    start_date = st.date_input("분석 시작일", datetime(2017, 1, 1))

    if st.button("비교 분석 실행"):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if comp_mode == "EPS 성장률 비교":
            # File 13: 정규화된 EPS 성장 추세
            for t in t_list:
                e_df, _, _, _ = fetch_stock_data(t)
                if e_df.empty: continue
                sub = e_df[e_df.index >= pd.to_datetime(start_date)]
                if sub.empty: continue
                norm_growth = (sub['EPS'] / sub['EPS'].iloc[0]) * 100
                ax.plot(norm_growth.index, norm_growth, marker='o', label=f"{t} (Last: {norm_growth.iloc[-1]:.0f})")
            ax.set_title("Normalized EPS Growth (Base = 100)")
            
        elif comp_mode == "섹터/지수 수익률":
            # File 12: 주가 성과 비교
            for t in t_list:
                p = yf.Ticker(t).history(start=start_date)['Close']
                if p.empty: continue
                norm_p = (p / p.iloc[0]) * 100
                ax.plot(norm_p.index, norm_p, label=f"{t} ({norm_p.iloc[-1]:.1f})")
            ax.set_title("Price Performance (Base = 100)")

        elif comp_mode == "상대 PER 추세":
            # File 9: PER 추세 비교
            for t in t_list:
                _, p_df, _, _ = fetch_stock_data(t)
                if p_df.empty: continue
                sub = p_df[p_df.index >= pd.to_datetime(start_date)]
                if sub.empty: continue
                norm_per = (sub['PER'] / sub['PER'].iloc[0]) * 100
                ax.plot(norm_per.index, norm_per, label=f"{t} (Current PER: {sub['PER'].iloc[-1]:.1f})")
            ax.set_title("Normalized PER Trend (Base = 100)")

        ax.axhline(100, color='black', lw=1, ls='--')
        ax.legend()
        st.pyplot(fig)

# ==========================================
# 메인 메뉴 관리
# ==========================================

def main():
    st.sidebar.title("🇺🇸 주식 분석 터미널")
    menu = st.sidebar.selectbox("메뉴 선택", ["홈", "개별 종목 밸류에이션", "종목 비교 분석"])
    
    if menu == "홈":
        st.title("Welcome to Investment Dashboard")
        st.markdown("""
        이 대시보드는 업로드하신 7개의 파이썬 분석 코드를 하나로 통합한 버전입니다.
        - **2017년 이후 데이터**만 참조하도록 설계되었습니다.
        - **Yahoo Finance**의 예측치(Estimates)와 **ChoiceStock**의 확정 실적을 결합합니다.
        - **배포 팁:** GitHub에 `app.py`와 `requirements.txt`만 있으면 바로 작동합니다.
        """)
    elif menu == "개별 종목 밸류에이션":
        run_single_valuation()
    elif menu == "종목 비교 분석":
        run_comparison()

if __name__ == "__main__":
    main()
