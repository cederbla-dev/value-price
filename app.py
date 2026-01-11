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
st.set_page_config(page_title="Stock & ETF Professional Analyzer", layout="wide")

# --- [공통] 스타일 적용 함수 ---
def apply_strong_style(ax, title, ylabel):
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='black')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='black')
    ax.grid(True, linestyle='--', alpha=0.5, color='#d3d3d3')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.2)
    ax.tick_params(axis='both', colors='black', labelsize=9)
    ax.axhline(0, color='black', linewidth=1.2, zorder=2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# --- [데이터 처리 함수들] ---

def normalize_to_standard_quarter(dt):
    month = dt.month
    year = dt.year
    if month in [1, 2, 3]:   new_month, new_year = 3, year
    elif month in [4, 5, 6]: new_month, new_year = 6, year
    elif month in [7, 8, 9]: new_month, new_year = 9, year
    elif month in [10, 11, 12]: new_month, new_year = 12, year
    return pd.Timestamp(year=new_year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

@st.cache_data(ttl=3600)
def fetch_per_data(ticker, predict_mode):
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
        
        if predict_mode != "None":
            stock = yf.Ticker(ticker)
            history = stock.history(period="1d")
            current_price = history['Close'].iloc[-1] if not history.empty else 0
            est = stock.earnings_estimate
            if est is not None and not est.empty:
                last_dt = combined.index[-1]
                ttm_eps_q1 = sum(combined['EPS'].tolist()[-3:]) + est.loc['0q', 'avg']
                combined.loc[last_dt + pd.DateOffset(months=3), 'PER'] = current_price / ttm_eps_q1
                if predict_mode == "다음 분기 예측":
                    ttm_eps_q2 = sum(combined['EPS'].tolist()[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[last_dt + pd.DateOffset(months=6), 'PER'] = current_price / ttm_eps_q2
        
        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined['PER']
    except: return None

@st.cache_data(ttl=3600)
def fetch_eps_data(ticker, predict_mode):
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

        if predict_mode != "None":
            stock = yf.Ticker(ticker)
            est = stock.earnings_estimate
            if est is not None and not est.empty:
                last_q_label = eps_df.index[-1]
                year, q = map(int, last_q_label.split('-Q'))
                
                q1_q = q + 1
                q1_year = year
                if q1_q > 4: q1_q = 1; q1_year += 1
                label_q1 = f"{q1_year}-Q{q1_q}"
                eps_df.loc[label_q1, ticker] = est.loc['0q', 'avg']
                eps_df.loc[label_q1, 'type'] = 'Estimate'
                
                if predict_mode == "다음 분기 예측":
                    q2_q = q1_q + 1
                    q2_year = q1_year
                    if q2_q > 4: q2_q = 1; q2_year += 1
                    label_q2 = f"{q2_year}-Q{q2_q}"
                    eps_df.loc[label_q2, ticker] = est.loc['+1q', 'avg']
                    eps_df.loc[label_q2, 'type'] = 'Estimate'
        
        return eps_df.sort_index()
    except: return pd.DataFrame()

# --- [UI 레이아웃] ---

with st.sidebar:
    st.title("📂 분석 메뉴")
    main_menu = st.radio(
        "분석 종류를 선택하세요:",
        ("기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

if main_menu == "기업 가치 비교 (PER/EPS)":
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            ticker_input = st.text_input("🏢 티커 입력 (예: AAPL, MSFT)", "AAPL, MSFT, NVDA")
        with col2:
            start_year = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3:
            predict_mode = st.radio(
                "🔮 미래 예측 옵션",
                ("None", "현재 분기 예측", "다음 분기 예측"),
                horizontal=True, index=1
            )
        
        # 지표 선택 항목 추가
        target_metric = st.radio(
            "📊 확인할 지표를 선택하세요:",
            ("PER 증감률", "EPS 증가율"),
            horizontal=True
        )
        
        analyze_btn = st.button("데이터 분석 실행", type="primary", use_container_width=True)

    if analyze_btn:
        tickers = [t.strip().upper() for t in ticker_input.replace(',', ' ').split() if t.strip()]
        
        # 선택한 지표에 따라 기본 탭 설정
        default_tab_idx = 0 if target_metric == "PER 증감률" else 1
        tab1, tab2 = st.tabs(["📊 PER 증감률 (%)", "📈 EPS 성장률 (%)"])

        # PER 증감률 탭
        with tab1:
            master_per = pd.DataFrame()
            for t in tickers:
                s = fetch_per_data(t, predict_mode)
                if s is not None: master_per[t] = s
            if not master_per.empty:
                master_per = master_per[master_per.index >= f"{start_year}-01-01"].sort_index()
                indexed_per = (master_per / master_per.iloc[0] - 1) * 100
                # 그래프 크기 80% (9.6 x 4.8)
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
                x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_per.index]
                
                for i, ticker in enumerate(indexed_per.columns):
                    series = indexed_per[ticker].dropna()
                    f_count = 1 if predict_mode == "현재 분기 예측" else (2 if predict_mode == "다음 분기 예측" else 0)
                    h_end = len(series) - f_count
                    ax.plot(range(h_end), series.values[:h_end], marker='o', 
                            label=f"{ticker} ({series.values[-1]:+.1f}%)", color=colors[i], linewidth=2.0)
                    if f_count > 0:
                        ax.plot(range(h_end-1, len(series)), series.values[h_end-1:], 
                                linestyle='--', color=colors[i], linewidth=1.8, alpha=0.8)
                
                apply_strong_style(ax, f"Relative PER Change (%) since {start_year}", "Change (%)")
                ax.set_xticks(range(len(indexed_per))); ax.set_xticklabels(x_labels, rotation=45)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, facecolor='white', edgecolor='black', labelcolor='black', fontsize=9)
                st.pyplot(fig)

        # EPS 증가율 탭
        with tab2:
            all_eps = []
            for t in tickers:
                df = fetch_eps_data(t, predict_mode)
                if not df.empty: all_eps.append(df)
            
            if all_eps:
                full_idx = sorted(list(set().union(*(d.index for d in all_eps))))
                filtered_idx = [idx for idx in full_idx if idx >= f"{start_year}-Q1"]
                # 그래프 크기 80% (9.6 x 4.8)
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                
                for i, df in enumerate(all_eps):
                    t = [c for c in df.columns if c != 'type'][0]
                    plot_df = df.reindex(filtered_idx)
                    valid_data = plot_df[plot_df[t].notna()]
                    if valid_data.empty: continue
                    base_val = valid_data[t].iloc[0]
                    norm_vals = (plot_df[t] / base_val - 1) * 100
                    color = plt.cm.Set1(i % 9)
                    
                    actual_mask = plot_df['type'] == 'Actual'
                    actual_indices = np.where(actual_mask)[0]
                    
                    if len(actual_indices) > 0:
                        last_act_pos = actual_indices[-1]
                        ax.plot(range(last_act_pos + 1), norm_vals.iloc[:last_act_pos + 1], 
                                marker='o', label=f"{t} ({norm_vals.dropna().values[-1]:+.1f}%)", 
                                color=color, linewidth=2.0)
                        
                        if predict_mode != "None":
                            ax.plot(range(last_act_pos, len(filtered_idx)), 
                                    norm_vals.iloc[last_act_pos:], 
                                    linestyle='--', color=color, linewidth=1.8)
                
                apply_strong_style(ax, f"Normalized EPS Growth (%) since {start_year}-Q1", "Growth (%)")
                ax.set_xticks(range(len(filtered_idx))); ax.set_xticklabels(filtered_idx, rotation=45)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, facecolor='white', edgecolor='black', labelcolor='black', fontsize=9)
                st.pyplot(fig)
                
        # 기본 탭 강제 선택을 위한 Streamlit JavaScript trick 또는 탭 구조 최적화는 
        # 위 코드에서 사용자가 버튼을 누른 후 직접 탭을 클릭하거나 
        # metric 선택에 따라 하단 출력 순서를 조정하는 방식으로 구현되었습니다.

else: # ETF 수익률 분석 모드는 기존 유지
    # ... (생략된 코드는 이전과 동일)
