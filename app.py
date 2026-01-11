import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import timedelta

# 기본 설정 및 경고 무시
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock Valuation & Growth Analyzer", layout="wide")

# --- 공통 함수 및 PER 관련 함수 ---

def normalize_to_standard_quarter(dt):
    """서로 다른 분기 마감일을 가장 가까운 표준 분기(3, 6, 9, 12월)로 조정"""
    month = dt.month
    year = dt.year
    if month in [1, 2, 3]:   new_month, new_year = 3, year
    elif month in [4, 5, 6]: new_month, new_year = 6, year
    elif month in [7, 8, 9]: new_month, new_year = 9, year
    elif month in [10, 11, 12]: new_month, new_year = 12, year
    return pd.Timestamp(year=new_year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

@st.cache_data(ttl=3600)
def fetch_multicycle_ticker_per(ticker, show_q1, show_q2):
    try:
        url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(io.StringIO(response.text))
        
        target_df = None
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains('PER').any():
                target_df = df.set_index(df.columns[0])
                break
        
        if target_df is None: return None

        per_raw = target_df[target_df.index.str.contains('PER')].transpose()
        eps_raw = target_df[target_df.index.str.contains('EPS')].transpose()
        
        combined = pd.DataFrame({
            'PER': pd.to_numeric(per_raw.iloc[:, 0], errors='coerce'),
            'EPS': pd.to_numeric(eps_raw.iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
        }).dropna()

        combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
        combined = combined.sort_index()
        historical_eps = combined['EPS'].tolist()
        
        if show_q1:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1d")
            current_price = history['Close'].iloc[-1] if not history.empty else 0
            est = stock.earnings_estimate
            if est is not None and not est.empty:
                last_dt = combined.index[-1]
                q1_dt = last_dt + pd.DateOffset(months=3)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1
                if show_q2:
                    q2_dt = q1_dt + pd.DateOffset(months=3)
                    ttm_eps_q2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[q2_dt, 'PER'] = current_price / ttm_eps_q2

        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined['PER']
    except:
        return None

# --- EPS 성장률 관련 함수 ---

def get_future_estimates_yf(ticker):
    try:
        stock = yf.Ticker(ticker)
        est = stock.earnings_estimate
        if est is not None and not est.empty:
            curr_est = est['avg'].iloc[0]
            next_est = est['avg'].iloc[1] if len(est) > 1 else None
            return {'current': curr_est, 'next': next_est}
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def _get_ticker_data_integrated(ticker):
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text), flavor='lxml')
        target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS').any()), None)
        if target_df is None: return pd.DataFrame()
        
        target_df = target_df.set_index(target_df.columns[0]).transpose()
        eps_df = target_df.iloc[:, [0]].copy()
        eps_df.columns = [ticker]
        eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
        eps_df = eps_df.dropna()
        
        def to_quarter_label(dt):
            actual_dt = (dt.replace(day=1) - timedelta(days=1)) if dt.day <= 5 else dt
            return f"{actual_dt.year}-Q{(actual_dt.month-1)//3 + 1}"

        eps_df.index = [to_quarter_label(d) for d in eps_df.index]
        eps_df[ticker] = pd.to_numeric(eps_df[ticker].astype(str).str.replace(',', ''), errors='coerce')
        eps_df = eps_df.groupby(level=0).last()
        eps_df['type'] = 'Actual'

        estimates = get_future_estimates_yf(ticker)
        if estimates:
            last_q = eps_df.index[-1]
            year, q = int(last_q.split('-Q')[0]), int(last_q.split('-Q')[1])
            for i, key in enumerate(['current', 'next'], 1):
                val = estimates[key]
                if val is not None:
                    new_q = q + i
                    new_year = year + (new_q - 1) // 4
                    actual_q = (new_q - 1) % 4 + 1
                    q_label = f"{new_year}-Q{actual_q}"
                    eps_df.loc[q_label, ticker] = val
                    eps_df.loc[q_label, 'type'] = 'Estimate'
        return eps_df
    except:
        return pd.DataFrame()

# --- 메인 UI 레이아웃 ---

st.title("📈 주식 가치 및 성장 통합 분석기")

with st.sidebar:
    st.header("🔍 설정 패널")
    ticker_input = st.text_input("분석 티커 입력 (예: AAPL, MSFT, TSLA)", "AAPL, MSFT, NVDA")
    start_year = st.number_input("기준 시작 연도", min_value=2010, max_value=2025, value=2020)
    
    st.markdown("---")
    st.subheader("PER 설정")
    show_q1 = st.checkbox("PER 현재 분기 예측 포함", value=True)
    show_q2 = st.checkbox("PER 다음 분기 예측 포함", value=False)
    
    analyze_btn = st.button("분석 실행", type="primary")

if analyze_btn:
    tickers = list(dict.fromkeys([t.strip().upper() for t in ticker_input.replace(',', ' ').split() if t.strip()]))
    
    tab1, tab2 = st.tabs(["📊 Relative PER Trend", "📈 EPS Growth Trend"])

    # --- Tab 1: PER 분석 ---
    with tab1:
        st.subheader("표준 분기 동기화 상대 PER 추이")
        master_per = pd.DataFrame()
        progress_per = st.progress(0)
        
        for idx, ticker in enumerate(tickers):
            series = fetch_multicycle_ticker_per(ticker, show_q1, show_q2)
            if series is not None:
                master_per[ticker] = series
            progress_per.progress((idx + 1) / len(tickers))
        
        if not master_per.empty:
            master_per = master_per[master_per.index >= f"{start_year}-01-01"].sort_index()
            if not master_per.empty and not master_per.iloc[0].isnull().any():
                indexed_per = (master_per / master_per.iloc[0]) * 100
                
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                ax.set_facecolor('white')
                
                x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_per.index]
                x_indices = np.arange(len(indexed_per))

                for ticker in indexed_per.columns:
                    series = indexed_per[ticker].dropna()
                    forecast_count = (1 if show_q1 else 0) + (1 if show_q2 else 0)
                    valid_indices = [indexed_per.index.get_loc(dt) for dt in series.index]
                    
                    if len(valid_indices) > forecast_count:
                        hist_idx = valid_indices[:-forecast_count] if forecast_count > 0 else valid_indices
                        hist_val = series.values[:-forecast_count] if forecast_count > 0 else series.values
                        line, = ax.plot(hist_idx, hist_val, marker='o', label=f"{ticker}", linewidth=2)
                        
                        if forecast_count > 0:
                            pred_idx = valid_indices[-forecast_count-1:]
                            pred_val = series.values[-forecast_count-1:]
                            ax.plot(pred_idx, pred_val, linestyle='--', color=line.get_color(), alpha=0.6)
                            ax.scatter(valid_indices[-forecast_count:], series.values[-forecast_count:], marker='D', s=50, color=line.get_color())

                ax.axhline(100, color='black', alpha=0.3, linestyle='--')
                ax.set_title(f"Relative PER Trend (Base: {start_year})", fontsize=14)
                ax.set_xticks(x_indices)
                ax.set_xticklabels(x_labels, rotation=45)
                ax.legend(loc='upper left', frameon=True)
                ax.grid(True, axis='y', alpha=0.3)
                st.pyplot(fig)
            else:
                st.warning("PER 데이터를 인덱스화할 수 없습니다 (시작 시점 데이터 부족).")

    # --- Tab 2: EPS 성장률 분석 ---
    with tab2:
        st.subheader("EPS 과거 실적 및 향후 성장률 비교")
        all_eps_data = []
        progress_eps = st.progress(0)
        
        for idx, ticker in enumerate(tickers):
            df = _get_ticker_data_integrated(ticker)
            if not df.empty:
                all_eps_data.append(df)
            progress_eps.progress((idx + 1) / len(tickers))

        if all_eps_data:
            combined_index = sorted(list(set().union(*(d.index for d in all_eps_data))))
            combined_index = [i for i in combined_index if i >= f"{start_year}-Q1"]
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            ax.set_facecolor('white')
            
            for df in all_eps_data:
                ticker = [c for c in df.columns if c != 'type'][0]
                base_data = df[df.index >= f"{start_year}-Q1"]
                if base_data.empty: continue
                base_val = base_data[ticker].dropna().iloc[0]
                
                plot_df = df.reindex(combined_index)
                norm_values = plot_df[ticker] / base_val
                
                actual_mask = plot_df['type'] == 'Actual'
                est_mask = plot_df['type'] == 'Estimate'
                
                if actual_mask.any():
                    x_actual = [combined_index.index(i) for i in plot_df[actual_mask].index]
                    line = ax.plot(x_actual, norm_values[actual_mask], marker='o', label=f"{ticker}", linewidth=2)
                    color = line[0].get_color()
                    
                    if est_mask.any():
                        last_actual_idx = plot_df[actual_mask].index[-1]
                        est_indices = [last_actual_idx] + list(plot_df[est_mask].index)
                        x_est = [combined_index.index(i) for i in est_indices]
                        ax.plot(x_est, norm_values[est_indices], marker='x', linestyle='--', color=color, alpha=0.7)

            ax.set_title(f"Normalized EPS Growth (Base: {start_year}-Q1)", fontsize=14)
            ax.set_xticks(range(len(combined_index)))
            ax.set_xticklabels(combined_index, rotation=45)
            ax.set_ylabel("Growth Factor")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.error("수집된 EPS 데이터가 없습니다.")

else:
    st.info("사이드바에서 티커를 입력하고 '분석 실행' 버튼을 눌러주세요.")
