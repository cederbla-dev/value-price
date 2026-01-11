import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import timedelta
import matplotlib.ticker as mtick

# 기본 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# --- 데이터 처리 함수 (기존 로직 유지) ---

def normalize_to_standard_quarter(dt):
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
        target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('PER').any()), None)
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
    except: return None

@st.cache_data(ttl=3600)
def fetch_ticker_eps_integrated(ticker):
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
        def to_q_label(dt):
            actual_dt = (dt.replace(day=1) - timedelta(days=1)) if dt.day <= 5 else dt
            return f"{actual_dt.year}-Q{(actual_dt.month-1)//3 + 1}"
        eps_df.index = [to_q_label(d) for d in eps_df.index]
        eps_df[ticker] = pd.to_numeric(eps_df[ticker].astype(str).str.replace(',', ''), errors='coerce')
        eps_df = eps_df.groupby(level=0).last()
        eps_df['type'] = 'Actual'
        stock = yf.Ticker(ticker)
        est = stock.earnings_estimate
        if est is not None and not est.empty:
            last_q = eps_df.index[-1]
            year, q = int(last_q.split('-Q')[0]), int(last_q.split('-Q')[1])
            for i, key in enumerate(['0q', '+1q']):
                if i < len(est):
                    val = est.loc[key, 'avg']
                    new_q = q + (i + 1)
                    q_label = f"{year + (new_q-1)//4}-Q{(new_q-1)%4 + 1}"
                    eps_df.loc[q_label, ticker], eps_df.loc[q_label, 'type'] = val, 'Estimate'
        return eps_df
    except: return pd.DataFrame()

# --- UI 레이아웃 ---

st.title("🚀 고시안성 주식 밸류에이션/성장 분석기")

with st.sidebar:
    st.header("⚙️ 설정")
    ticker_input = st.text_input("티커 입력", "AAPL, MSFT, NVDA, TSLA")
    start_year = st.number_input("기준 연도", 2010, 2025, 2020)
    st.markdown("---")
    ans1 = st.checkbox("PER 현재분기 예측 포함", True)
    ans2 = st.checkbox("PER 다음분기 예측 포함", False)
    analyze_btn = st.button("데이터 분석 시작", type="primary")

if analyze_btn:
    tickers = [t.strip().upper() for t in ticker_input.replace(',', ' ').split() if t.strip()]
    tab1, tab2 = st.tabs(["📊 PER 증감률 (%)", "📈 EPS 성장률 (%)"])

    # 공통 그래프 스타일 설정 함수
    def apply_strong_style(ax, title, ylabel):
        ax.set_facecolor('white')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, color='#d3d3d3')
        # 축 선 강화
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.axhline(0, color='black', linewidth=2, zorder=2) # 0% 기준선
        ax.yaxis.set_major_formatter(mtick.PercentFormatter()) # % 단위 표시

    # --- Tab 1: PER (%) ---
    with tab1:
        master_per = pd.DataFrame()
        for t in tickers:
            s = fetch_multicycle_ticker_per(t, ans1, ans2)
            if s is not None: master_per[t] = s
        
        if not master_per.empty:
            master_per = master_per[master_per.index >= f"{start_year}-01-01"].sort_index()
            # 기준점 대비 % 변환: (현재값 / 기준값 - 1) * 100
            indexed_per = (master_per / master_per.iloc[0] - 1) * 100
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            colors = plt.cm.tab10(np.linspace(0, 1, len(tickers))) # 강한 색감 테마
            
            x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_per.index]
            for i, ticker in enumerate(indexed_per.columns):
                series = indexed_per[ticker].dropna()
                f_count = (1 if ans1 else 0) + (1 if ans2 else 0)
                v_idx = [indexed_per.index.get_loc(dt) for dt in series.index]
                
                # 실선 (과거)
                h_idx = v_idx[:-f_count] if f_count > 0 else v_idx
                ax.plot(h_idx, series.values[:-f_count] if f_count > 0 else series.values, 
                        marker='o', label=ticker, linewidth=2.5, color=colors[i], markersize=8)
                # 점선 (예측)
                if f_count > 0:
                    p_idx = v_idx[-f_count-1:]
                    ax.plot(p_idx, series.values[-f_count-1:], linestyle='--', color=colors[i], linewidth=2, alpha=0.8)
                    ax.scatter(v_idx[-f_count:], series.values[-f_count:], marker='D', s=80, color=colors[i], edgecolors='white', zorder=5)

            apply_strong_style(ax, f"PER Relative Change (%) since {start_year}", "Change (%)")
            ax.set_xticks(range(len(indexed_per)))
            ax.set_xticklabels(x_labels, rotation=45)
            ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
            st.pyplot(fig)

    # --- Tab 2: EPS (%) ---
    with tab2:
        all_eps = []
        for t in tickers:
            df = fetch_ticker_eps_integrated(t)
            if not df.empty: all_eps.append(df)
        
        if all_eps:
            c_idx = sorted(list(set().union(*(d.index for d in all_eps))))
            c_idx = [i for i in c_idx if i >= f"{start_year}-Q1"]
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            for i, df in enumerate(all_eps):
                t = [c for c in df.columns if c != 'type'][0]
                base_data = df[df.index >= f"{start_year}-Q1"]
                if base_data.empty: continue
                base_val = base_data[t].dropna().iloc[0]
                
                plot_df = df.reindex(c_idx)
                # % 변환: (값 / 기준값 - 1) * 100
                norm_vals = (plot_df[t] / base_val - 1) * 100
                
                act_m, est_m = plot_df['type'] == 'Actual', plot_df['type'] == 'Estimate'
                color = plt.cm.Set1(i % 9)
                
                if act_m.any():
                    x_act = [c_idx.index(idx) for idx in plot_df[act_m].index]
                    ax.plot(x_act, norm_vals[act_m], marker='o', label=t, linewidth=2.5, color=color, markersize=8)
                    if est_m.any():
                        last_act = plot_df[act_m].index[-1]
                        e_indices = [last_act] + list(plot_df[est_m].index)
                        x_est = [c_idx.index(idx) for idx in e_indices]
                        ax.plot(x_est, norm_vals[e_indices], marker='x', linestyle='--', color=color, linewidth=2)

            apply_strong_style(ax, f"EPS Growth (%) since {start_year}-Q1", "Growth (%)")
            ax.set_xticks(range(len(c_idx)))
            ax.set_xticklabels(c_idx, rotation=45)
            ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
            st.pyplot(fig)
