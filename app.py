import streamlit as st
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import numpy as np
import warnings

# -----------------------------------------------------------
# [0] 환경 설정 및 공통 유틸리티
# -----------------------------------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="미국주식 통합 분석 시스템", layout="wide")
plt.style.use('seaborn-v0_8-whitegrid')

def fmt(val):
    try: return "{:.2f}".format(float(val))
    except: return str(val)

def format_df(df):
    return df.map(lambda x: fmt(x) if isinstance(x, (int, float)) else x)

def normalize_to_standard_quarter(dt):
    """서로 다른 분기 마감일을 표준 분기(3, 6, 9, 12월)로 조정"""
    month, year = dt.month, dt.year
    if month in [1, 2, 3]:   new_month = 3
    elif month in [4, 5, 6]: new_month = 6
    elif month in [7, 8, 9]: new_month = 9
    else:                    new_month = 12
    return pd.Timestamp(year=year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

# -----------------------------------------------------------
# [Module 1] 개별 종목 밸류에이션 (기존 기능)
# -----------------------------------------------------------
def run_single_valuation():
    st.header("💎 개별 종목 밸류에이션")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: ticker = st.text_input("티커 입력", "TSLA").upper().strip()
    with col2: base_year_input = st.selectbox("기준 연도", range(2017, 2026), index=0)
    with col3: include_est = st.radio("예측치 포함", ["None", "Current Q", "Next Q"], horizontal=True)

    if ticker:
        try:
            url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            dfs = pd.read_html(io.StringIO(response.text))
            eps_df_raw = pd.DataFrame()
            for df in dfs:
                if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                    target = df.set_index(df.columns[0]).transpose()
                    eps_df_raw = target.iloc[:, [0]].copy()
                    eps_df_raw.columns = ['EPS']
                    break
            eps_df_raw.index = pd.to_datetime(eps_df_raw.index, format='%y.%m.%d', errors='coerce')
            eps_df_raw = eps_df_raw.dropna().sort_index()
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            tab1, tab2 = st.tabs(["📉 연도별 시뮬레이션", "📊 4분기 실적 기반 분석"])
            with tab1:
                combined = eps_df_raw.copy()
                combined.index = combined.index.strftime('%Y-%m')
                price_m = stock.history(start="2017-01-01", interval="1mo")['Close']
                price_m.index = price_m.index.tz_localize(None).strftime('%Y-%m')
                combined = pd.merge(combined, price_m, left_index=True, right_index=True, how='inner')
                summary_data = []
                for by in range(2017, 2026):
                    df_p = combined[combined.index >= f'{by}-01'].copy()
                    if df_p.empty or df_p.iloc[0]['EPS'] <= 0: continue
                    sf = df_p.iloc[0]['Close'] / df_p.iloc[0]['EPS']
                    df_p['Fair'] = df_p['EPS'] * sf
                    gap = ((current_price - df_p['Fair'].iloc[-1]) / df_p['Fair'].iloc[-1]) * 100
                    summary_data.append({"기준년도": by, "PER": sf, "적정가": df_p['Fair'].iloc[-1], "현재가": current_price, "괴리율": gap})
                    if by == base_year_input:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(df_p.index, df_p['Close'], label='Market')
                        ax.plot(df_p.index, df_p['Fair'], label='Fair', ls='--')
                        plt.xticks(rotation=45); st.pyplot(fig)
                st.table(format_df(pd.DataFrame(summary_data)))
            with tab2:
                est = stock.earnings_estimate
                target_eps = eps_df_raw['EPS'].iloc[-3:].sum() + (est['avg'].iloc[0] if est is not None else 0)
                res10 = []
                for i in range(0, len(eps_df_raw)-3, 4):
                    grp = eps_df_raw.iloc[i:i+4]
                    e_sum = grp['EPS'].sum()
                    per = grp['EPS'].mean() # 단순 예시 로직
                    fair = target_eps * (current_price/e_sum) # 원본 로직 참조
                    res10.append({"기간": f"{grp.index[0].year}", "PER": e_sum, "적정가": fair})
                st.table(format_df(pd.DataFrame(res10)))
        except Exception as e: st.error(f"오류: {e}")

# -----------------------------------------------------------
# [Module 2] 종목 비교 분석 (PER 및 EPS 성장률 동기화)
# -----------------------------------------------------------
def fetch_comp_data_sync(ticker, include_mode, metric_type="PER"):
    """PER 또는 EPS 데이터를 가져와 표준 분기로 동기화 및 예측치 추가"""
    try:
        url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        
        target_df = None
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains('PER').any():
                target_df = df.set_index(df.columns[0])
                break
        if target_df is None: return None

        per_raw = pd.to_numeric(target_df[target_df.index.str.contains('PER')].transpose().iloc[:, 0], errors='coerce')
        eps_raw = pd.to_numeric(target_df[target_df.index.str.contains('EPS')].transpose().iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
        
        combined = pd.DataFrame({'PER': per_raw, 'EPS': eps_raw}).dropna()
        combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
        combined = combined.sort_index()
        
        # 예측치 계산용 (TTM 슬라이딩 윈도우)
        if include_mode != "None":
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            est = stock.earnings_estimate
            historical_eps = combined['EPS'].tolist()
            
            if est is not None and not est.empty:
                # Current Q
                q1_dt = combined.index[-1] + pd.DateOffset(months=3)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1
                combined.loc[q1_dt, 'EPS'] = ttm_eps_q1 # EPS 성장률용으로 TTM EPS 저장
                combined.loc[q1_dt, 'is_est'] = True
                
                # Next Q
                if include_mode == "Next Q":
                    q2_dt = q1_dt + pd.DateOffset(months=3)
                    ttm_eps_q2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[q2_dt, 'PER'] = current_price / ttm_eps_q2
                    combined.loc[q2_dt, 'EPS'] = ttm_eps_q2
                    combined.loc[q2_dt, 'is_est'] = True

        combined['is_est'] = combined['is_est'].fillna(False)
        # 표준 분기 스냅
        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        
        return combined[[metric_type, 'is_est']]
    except: return None

def run_comparison():
    st.header("⚖️ 종목 간 지표 비교 (Quarter Sync & Forecast)")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tickers_input = st.text_input("비교 티커 (쉼표 구분)", "SNPS, FDS, GOOGL")
        t_list = [x.strip().upper() for x in tickers_input.replace(',', ' ').split() if x.strip()]
    with col2:
        comp_mode = st.selectbox("비교 지표 선택", ["PER 추세", "EPS 성장률"])
    with col3:
        include_mode = st.radio("예측치 선택", ["None", "Current Q", "Next Q"], horizontal=True)

    start_year = st.number_input("분석 시작 연도", 2010, 2025, 2020)

    if st.button("비교 차트 생성"):
        metric = "PER" if comp_mode == "PER 추세" else "EPS"
        master_list = []
        
        for t in t_list:
            data = fetch_comp_data_sync(t, include_mode, metric)
            if data is not None:
                data.columns = [t, f"{t}_is_est"]
                master_list.append(data)
        
        if not master_list:
            st.error("데이터를 불러오지 못했습니다."); return

        # 데이터 통합
        combined_df = pd.concat(master_list, axis=1)
        combined_df = combined_df[combined_df.index >= f"{start_year}-01-01"].sort_index()
        
        # 기준점 100으로 정규화 (Base 100)
        indexed_df = pd.DataFrame(index=combined_df.index)
        for t in t_list:
            if t in combined_df.columns:
                base_val = combined_df[t].dropna().iloc[0]
                indexed_df[t] = (combined_df[t] / base_val) * 100
                indexed_df[f"{t}_is_est"] = combined_df[f"{t}_is_est"]

        fig, ax = plt.subplots(figsize=(12, 6))
        x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_df.index]
        
        for t in t_list:
            if t not in indexed_df.columns: continue
            
            series = indexed_df[t].dropna()
            is_est_series = indexed_df[f"{t}_is_est"].reindex(series.index).fillna(False)
            
            # 최종 성장률 % 계산
            final_growth = series.iloc[-1] - 100
            label_text = f"{t} (Actual) {final_growth:+.1f}%"
            
            # 실제 데이터와 예측 데이터 분리 추출
            actual_idx = [indexed_df.index.get_loc(d) for d in series[~is_est_series].index]
            actual_val = series[~is_est_series].values
            
            line, = ax.plot(actual_idx, actual_val, marker='o', label=label_text, linewidth=2)
            
            # 예측치 연결 (실제 마지막 데이터부터 예측 데이터까지 점선)
            if is_est_series.any():
                est_part = series[is_est_series]
                # 연결을 위해 실제 데이터의 마지막 포인트를 포함
                last_actual_date = series[~is_est_series].index[-1]
                connect_dates = [last_actual_date] + est_part.index.tolist()
                
                connect_idx = [indexed_df.index.get_loc(d) for d in connect_dates]
                connect_val = series.loc[connect_dates].values
                
                ax.plot(connect_idx, connect_val, ls='--', marker='D', color=line.get_color(), alpha=0.7)

        ax.set_xticks(range(len(indexed_df)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.axhline(100, color='black', alpha=0.5, ls='-')
        ax.set_title(f"Comparison: {comp_mode} (Base 100 at {start_year})")
        ax.set_ylabel("Normalized Value")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

# -----------------------------------------------------------
# [Module 3] 섹터 수익률 (기존 기능)
# -----------------------------------------------------------
def run_sector_perf():
    st.header("📊 섹터 수익률 분석")
    selected = st.multiselect("ETF 선택", ["SPY", "QQQ", "XLK", "XLY", "XLF", "XLV"], default=["SPY", "QQQ", "XLK"])
    start_date = st.date_input("시작 날짜", datetime(2023, 1, 1))
    if st.button("수익률 확인"):
        prices = pd.DataFrame()
        for t in selected:
            prices[t] = yf.Ticker(t).history(start=start_date)['Close']
        if not prices.empty:
            norm = (prices / prices.iloc[0]) * 100
            fig, ax = plt.subplots(figsize=(10, 5))
            for c in norm.columns: ax.plot(norm.index, norm[c], label=c)
            ax.axhline(100, color='black', ls='--')
            ax.legend(); st.pyplot(fig)

# -----------------------------------------------------------
# [Main]
# -----------------------------------------------------------
def main():
    st.sidebar.title("🇺🇸 주식 분석 터미널")
    menu = st.sidebar.radio("메뉴", ["홈", "개별 종목 밸류에이션", "종목 비교 분석", "섹터 수익률"])
    if menu == "홈":
        st.title("US Stock Analytics v3")
        st.info("PER 및 EPS 비교 시 결산월 자동 동기화 기능이 적용되었습니다.")
    elif menu == "개별 종목 밸류에이션": run_single_valuation()
    elif menu == "종목 비교 분석": run_comparison()
    elif menu == "섹터 수익률": run_sector_perf()

if __name__ == "__main__":
    main()
