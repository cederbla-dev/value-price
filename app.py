import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import io
import numpy as np
from datetime import datetime
import warnings

# 환경 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="미국주식 통합 분석 시스템", layout="wide")
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# [Core] 데이터 수집 엔진 (캐싱 적용)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(res.text))
        eps_raw, per_raw = pd.DataFrame(), pd.DataFrame()
        
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                temp = df.set_index(df.columns[0]).filter(like='EPS', axis=0).transpose()
                temp.index = pd.to_datetime(temp.index, format='%y.%m.%d', errors='coerce')
                eps_raw = temp.dropna().sort_index(); eps_raw.columns = ['EPS']
            if df.iloc[:, 0].astype(str).str.contains('PER').any():
                temp = df.set_index(df.columns[0]).filter(like='PER', axis=0).transpose()
                temp.index = pd.to_datetime(temp.index, format='%y.%m.%d', errors='coerce')
                per_raw = temp.dropna().sort_index(); per_raw.columns = ['PER']
        
        stock = yf.Ticker(ticker)
        price = stock.history(start="2016-10-01")['Close']
        if price.index.tz is not None: price.index = price.index.tz_localize(None)
        
        est = stock.earnings_estimate
        est_dict = {}
        if est is not None and not est.empty:
            est_dict = {
                'curr_q': est.loc['0q', 'avg'] if '0q' in est.index else None,
                'next_q': est.loc['+1q', 'avg'] if '+1q' in est.index else None
            }
        return eps_raw, per_raw, price, est_dict
    except:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), {}

# ==========================================
# [Module 1] 개별 종목 밸류에이션
# ==========================================

def run_single_valuation():
    st.header("💎 종목별 밸류에이션 및 적정주가")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: ticker = st.text_input("티커 입력", "AAPL").upper().strip()
    with col2: base_year = st.number_input("기준 시작 연도", 2017, 2025, 2017)
    with col3: include_est = st.radio("예측치 포함 (개별)", ["None", "Current Q", "Next Q"], horizontal=True)

    if not ticker: return
    eps_df, per_df, price_ser, ests = fetch_stock_data(ticker)
    if eps_df.empty: st.error("데이터를 찾을 수 없습니다."); return

    eps_combined = eps_df[eps_df.index >= f"{base_year}-01-01"].copy()
    if include_est != "None" and ests:
        last_dt = eps_combined.index[-1]
        if ests['curr_q']: eps_combined.loc[last_dt + pd.DateOffset(months=3)] = [ests['curr_q']]
        if include_est == "Next Q" and ests['next_q']: eps_combined.loc[last_dt + pd.DateOffset(months=6)] = [ests['next_q']]

    tab1, tab2 = st.tabs(["📊 적정주가 시뮬레이션", "📈 PER 밴드 및 통계"])
    with tab1:
        price_m = price_ser.resample('M').last()
        df_val = eps_combined.join(price_m, how='left')
        df_val['Close'] = df_val['Close'].ffill()
        
        summary_rows = []
        current_price = price_ser.iloc[-1]
        for yr in range(base_year, datetime.now().year + 1):
            subset = df_val[df_val.index >= f"{yr}-01-01"]
            if len(subset) < 2 or subset.iloc[0]['EPS'] <= 0: continue
            mult = subset.iloc[0]['Close'] / subset.iloc[0]['EPS']
            fair_val = subset['EPS'].iloc[-1] * mult
            gap = ((current_price - fair_val) / fair_val) * 100
            summary_rows.append({"기준연도": yr, "기준PER": f"{mult:.1f}x", "Target EPS": f"${subset['EPS'].iloc[-1]:.2f}", "적정주가": round(fair_val, 2), "괴리율": f"{gap:+.2f}%", "판단": "고평가" if gap > 0 else "저평가"})
        st.table(pd.DataFrame(summary_rows))
        
        if summary_rows:
            fig, ax = plt.subplots(figsize=(12, 5))
            mult_first = float(summary_rows[0]['기준PER'].replace('x',''))
            df_plot = df_val[df_val.index >= f"{summary_rows[0]['기준연도']}-01-01"].copy()
            df_plot['Fair'] = df_plot['EPS'] * mult_first
            ax.plot(df_plot.index, df_plot['Close'], label='Price', marker='o'); ax.plot(df_plot.index, df_plot['Fair'], label='Fair', ls='--')
            ax.legend(); st.pyplot(fig)

    with tab2:
        p_sub = per_df[per_df.index >= f"{base_year}-01-01"]
        if not p_sub.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(p_sub.index, p_sub['PER'], marker='o'); ax.axhline(p_sub['PER'].mean(), color='red', ls='--', label='Mean')
            ax.legend(); st.pyplot(fig)

# ==========================================
# [Module 2] 종목 비교 분석 (EPS/PER 예측 반영)
# ==========================================

def run_comparison():
    st.header("⚖️ 종목 간 지표 비교")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_input("비교 티커 (쉼표 구분)", "AAPL, MSFT, GOOGL, NVDA")
        t_list = [x.strip().upper() for x in tickers_input.split(',')]
    with col2:
        include_est_comp = st.radio("예측치 포함 (비교)", ["None", "Current Q", "Next Q"], horizontal=True)

    comp_mode = st.selectbox("비교 지표 선택", ["EPS 성장률 비교", "상대 PER 추세"])
    start_date = st.date_input("분석 시작일", datetime(2017, 1, 1))

    if st.button("비교 분석 실행"):
        fig, ax = plt.subplots(figsize=(12, 6))
        for t in t_list:
            e_df, p_df, price_ser, ests = fetch_stock_data(t)
            if e_df.empty or p_df.empty or price_ser.empty: continue
            
            working_eps = e_df[e_df.index >= pd.to_datetime(start_date)].copy()
            working_per = p_df[p_df.index >= pd.to_datetime(start_date)].copy()
            current_price = price_ser.iloc[-1]
            
            # 예측치 반영 로직 (EPS 및 PER 동시 계산)
            if include_est_comp != "None" and ests:
                last_dt = working_eps.index[-1]
                if ests.get('curr_q'):
                    working_eps.loc[last_dt + pd.DateOffset(months=3)] = [ests['curr_q']]
                    fwd_per = current_price / ests['curr_q'] if ests['curr_q'] > 0 else np.nan
                    working_per.loc[last_dt + pd.DateOffset(months=3)] = [fwd_per]
                if include_est_comp == "Next Q" and ests.get('next_q'):
                    working_eps.loc[last_dt + pd.DateOffset(months=6)] = [ests['next_q']]
                    fwd_per = current_price / ests['next_q'] if ests['next_q'] > 0 else np.nan
                    working_per.loc[last_dt + pd.DateOffset(months=6)] = [fwd_per]

            display_df = working_eps if comp_mode == "EPS 성장률 비교" else working_per
            col_name = 'EPS' if comp_mode == "EPS 성장률 비교" else 'PER'
            
            # 정규화 및 그래프 그리기
            norm_series = (display_df[col_name] / display_df[col_name].dropna().iloc[0]) * 100
            actual_len = len(e_df[e_df.index >= pd.to_datetime(start_date)])
            
            line, = ax.plot(norm_series.iloc[:actual_len].index, norm_series.iloc[:actual_len], marker='o', label=t)
            if len(norm_series) > actual_len:
                ax.plot(norm_series.iloc[actual_len-1:].index, norm_series.iloc[actual_len-1:], ls='--', marker='x', color=line.get_color())

        ax.axhline(100, color='black', lw=1, ls='--')
        ax.set_title(f"Normalized {comp_mode} (Base=100)")
        ax.legend(); st.pyplot(fig)

# ==========================================
# [Module 3] 섹터 및 지수 수익률
# ==========================================

def run_sector_perf():
    st.header("📊 섹터 및 지수 수익률 분석")
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
# 메인 제어부
# ==========================================

def main():
    st.sidebar.title("🇺🇸 주식 분석 터미널")
    menu = st.sidebar.selectbox("메뉴 선택", ["홈", "개별 종목 밸류에이션", "종목 비교 분석 (EPS/PER)", "섹터/지수 수익률"])
    
    if menu == "홈":
        st.title("US Stock Analysis System")
        st.info("왼쪽 메뉴에서 분석 도구를 선택하세요. 모든 데이터는 2017년 이후를 기준으로 정규화됩니다.")
    elif menu == "개별 종목 밸류에이션": run_single_valuation()
    elif menu == "종목 비교 분석 (EPS/PER)": run_comparison()
    elif menu == "섹터/지수 수익률": run_sector_perf()

if __name__ == "__main__":
    main()
