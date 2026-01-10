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

# 공통 유틸리티: 소수점 2자리 반올림
def format_val(val):
    try:
        return round(float(val), 2)
    except:
        return val

# ==========================================
# [Shared] 데이터 수집 및 동기화 함수
# ==========================================

def normalize_to_standard_quarter(dt):
    month, year = dt.month, dt.year
    if month in [1, 2, 3]:   new_month = 3
    elif month in [4, 5, 6]: new_month = 6
    elif month in [7, 8, 9]: new_month = 9
    else:                    new_month = 12
    return pd.Timestamp(year=year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

@st.cache_data(ttl=3600)
def fetch_ticker_full_data(ticker, show_q1, show_q2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('PER').any()), None)
        if target_df is None: return None, None
        
        per_raw = pd.to_numeric(target_df[target_df.index.str.contains('PER')].transpose().iloc[:, 0], errors='coerce')
        eps_raw = pd.to_numeric(target_df[target_df.index.str.contains('EPS')].transpose().iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
        combined = pd.DataFrame({'PER': per_raw, 'EPS': eps_raw}).dropna()
        combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
        combined = combined.sort_index()
        
        if show_q1:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            est = stock.earnings_estimate
            if est is not None and not est.empty:
                historical_eps = combined['EPS'].tolist()
                q1_dt = combined.index[-1] + pd.DateOffset(months=3)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1
                if show_q2:
                    q2_dt = q1_dt + pd.DateOffset(months=3)
                    ttm_eps_q2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[q2_dt, 'PER'] = current_price / ttm_eps_q2

        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined['PER'], combined['EPS']
    except: return None, None

# ==========================================
# [Module 1] 개별 종목 밸류에이션
# ==========================================

def run_single_valuation():
    st.header("💎 종목별 밸류에이션 (개별)")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: ticker = st.text_input("티커 입력", "AAPL").upper().strip()
    with col2: base_year = st.number_input("기준 연도", 2017, 2025, 2017)
    with col3: include_est = st.radio("예측치 포함", ["None", "Current Q", "Next Q"], horizontal=True)

    if ticker:
        q1, q2 = (include_est in ["Current Q", "Next Q"]), (include_est == "Next Q")
        per_s, _ = fetch_ticker_full_data(ticker, q1, q2)
        if per_s is not None:
            plot_df = per_s[per_s.index >= f"{base_year}-01-01"]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(plot_df.index, plot_df, marker='o', label=f"{ticker} PER")
            mean_val = plot_df.mean()
            ax.axhline(mean_val, color='red', ls='--', label=f'Mean: {mean_val:.2f}')
            ax.legend(); st.pyplot(fig)

# ==========================================
# [Module 2] 종목 비교 분석 (Sync)
# ==========================================

def run_comparison():
    st.header("⚖️ 종목 간 지표 비교 (Sync)")
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_input("비교 티커 (쉼표 구분)", "AAPL, MSFT, AVGO, NVDA")
        t_list = [x.strip().upper() for x in tickers_input.split(',')]
    with col2:
        include_est_comp = st.radio("예측치 포함 (비교)", ["None", "Current Q", "Next Q"], horizontal=True)

    comp_mode = st.selectbox("비교 지표 선택", ["상대 PER 추세", "EPS 성장률 비교"])
    start_year = st.number_input("분석 시작 연도", 2010, 2025, 2017)

    if st.button("비교 분석 실행"):
        q1, q2 = (include_est_comp in ["Current Q", "Next Q"]), (include_est_comp == "Next Q")
        master_df = pd.DataFrame()
        for t in t_list:
            per_s, eps_s = fetch_ticker_full_data(t, q1, q2)
            if per_s is not None:
                master_df[t] = per_s if comp_mode == "상대 PER 추세" else eps_s

        if not master_df.empty:
            master_df = master_df[master_df.index >= f"{start_year}-01-01"].sort_index()
            indexed_df = (master_df / master_df.iloc[0]) * 100
            fig, ax = plt.subplots(figsize=(15, 8))
            x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_df.index]
            for ticker in indexed_df.columns:
                series = indexed_df[ticker].dropna()
                valid_indices = [indexed_df.index.get_loc(dt) for dt in series.index]
                forecast_count = (1 if q1 else 0) + (1 if q2 else 0)
                
                # 범례 숫자 소수점 2자리 적용
                label_val = f"{ticker} ({series.iloc[-1]:.2f})"
                
                if forecast_count > 0:
                    ax.plot(valid_indices[:-forecast_count], series.values[:-forecast_count], marker='o', label=label_val)
                    ax.plot(valid_indices[-forecast_count-1:], series.values[-forecast_count-1:], ls='--', marker='x', alpha=0.7)
                else:
                    ax.plot(valid_indices, series.values, marker='o', label=label_val)
            ax.set_xticks(range(len(indexed_df))); ax.set_xticklabels(x_labels, rotation=45)
            ax.axhline(100, color='black', alpha=0.5); ax.legend(); st.pyplot(fig)

# ==========================================
# [Module 3] 섹터 수익률 (포맷팅 강화)
# ==========================================

def run_sector_perf():
    st.header("📊 섹터 및 지수 수익률 분석 (분기 기준)")
    
    all_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
    selected = st.multiselect("분석할 ETF 선택", all_tickers, default=["SPY", "QQQ", "XLK"])
    
    col1, col2 = st.columns(2)
    with col1:
        sel_year = st.selectbox("시작 연도", range(2017, datetime.now().year + 1))
    with col2:
        sel_quarter = st.selectbox("시작 분기", [1, 2, 3, 4])
    
    q_map = {1: "-01-01", 2: "-04-01", 3: "-07-01", 4: "-10-01"}
    start_date_str = f"{sel_year}{q_map[sel_quarter]}"

    if st.button("수익률 차트 생성"):
        combined_price = pd.DataFrame()
        for t in selected:
            df = yf.Ticker(t).history(start="2017-01-01", interval="1mo", auto_adjust=True)
            if not df.empty:
                df.index = df.index.strftime('%Y-%m-%d')
                combined_price[t] = df['Close']
        
        if not combined_price.empty:
            available_dates = combined_price.index[combined_price.index >= start_date_str]
            if len(available_dates) == 0:
                st.error("해당 시점 이후의 데이터가 없습니다."); return
            
            base_date = available_dates[0]
            norm_df = (combined_price.loc[base_date:] / combined_price.loc[base_date]) * 100
            
            fig, ax = plt.subplots(figsize=(15, 8))
            last_val_idx = norm_df.iloc[-1].sort_values(ascending=False)
            
            for ticker in last_val_idx.index:
                lw = 4 if ticker in ["SPY", "QQQ"] else 2
                zo = 5 if ticker in ["SPY", "QQQ"] else 2
                ax.plot(norm_df.index, norm_df[ticker], label=f"{ticker} ({last_val_idx[ticker]:.2f})", linewidth=lw, zorder=zo)
            
            q_ticks = [d for d in norm_df.index if d.endswith(('-01-01', '-04-01', '-07-01', '-10-01'))]
            ax.set_xticks(q_ticks if q_ticks else norm_df.index[::3])
            plt.xticks(rotation=45)
            ax.axhline(100, color='black', ls='--')
            ax.set_title(f"ETF Performance (Base: {base_date} = 100)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
            # 테이블 출력 부분 (소수점 2자리 강제 문자열 포맷팅)
            st.write(f"### 🏆 {base_date} 이후 누적 수익률 (%)")
            performance_pct = (last_val_idx - 100).to_frame(name="수익률 (%)")
            
            # map을 사용하여 소수점 둘째 자리 문자열로 변환 (st.table 자동 포맷팅 방지)
            performance_pct["수익률 (%)"] = performance_pct["수익률 (%)"].map('{:.2f}'.format)
            
            st.table(performance_pct)

# ==========================================
# 메인 메뉴
# ==========================================

def main():
    st.sidebar.title("🇺🇸 주식 분석 터미널")
    menu = st.sidebar.selectbox("메뉴 선택", ["홈", "개별 종목 밸류에이션", "종목 비교 분석 (Sync)", "섹터/지수 수익률"])
    if menu == "홈":
        st.title("US Stock Analysis System")
        st.info("모든 수치는 소수점 둘째 자리까지 표시됩니다.")
    elif menu == "개별 종목 밸류에이션": run_single_valuation()
    elif menu == "종목 비교 분석 (Sync)": run_comparison()
    elif menu == "섹터/지수 수익률": run_sector_perf()

if __name__ == "__main__":
    main()
