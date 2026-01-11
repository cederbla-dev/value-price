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
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='black')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', color='black')
    ax.grid(True, linestyle='--', alpha=0.5, color='#d3d3d3')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis='both', colors='black', labelsize=8)
    ax.axhline(0, color='black', linewidth=1.5, zorder=2)

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
def fetch_valuation_data(ticker, predict_mode):
    try:
        # 1. 확정 실적(EPS) 수집
        url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        dfs = pd.read_html(io.StringIO(response.text))
        
        eps_df = pd.DataFrame()
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                target = df.set_index(df.columns[0]).transpose()
                eps_df = target.iloc[:, [0]].copy()
                eps_df.columns = ['EPS']
                break
        
        if eps_df.empty: return None

        eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
        eps_df = eps_df.dropna()
        
        def adjust_date(dt):
            return (dt.replace(day=1) - timedelta(days=1)).strftime('%Y-%m') if dt.day <= 5 else dt.strftime('%Y-%m')
        
        eps_df.index = [adjust_date(d) for d in eps_df.index]
        eps_df['EPS'] = pd.to_numeric(eps_df['EPS'].astype(str).str.replace(',', ''), errors='coerce')

        # 2. 주가 데이터 수집
        stock = yf.Ticker(ticker)
        price_df = stock.history(start="2017-01-01", interval="1mo", auto_adjust=False)
        if price_df.index.tz is not None: price_df.index = price_df.index.tz_localize(None)
        price_df.index = price_df.index.strftime('%Y-%m')
        price_df = price_df[['Close']].copy()
        price_df = price_df[~price_df.index.duplicated(keep='last')]

        combined = pd.merge(eps_df, price_df, left_index=True, right_index=True, how='inner')
        combined = combined.sort_index(ascending=True)

        # 3. 예측 데이터 추가
        if predict_mode != "None":
            est = stock.earnings_estimate
            current_price = stock.fast_info['last_price'] if 'last_price' in stock.fast_info else price_df['Close'].iloc[-1]
            if est is not None and not est.empty:
                last_date_obj = pd.to_datetime(combined.index[-1])
                curr_val = est['avg'].iloc[0]
                date_curr = (last_date_obj + pd.DateOffset(months=3)).strftime('%Y-%m')
                combined.loc[f"{date_curr} (Est.)"] = [curr_val, current_price]
                
                if predict_mode == "다음 분기 예측" and len(est) > 1:
                    next_val = est['avg'].iloc[1]
                    date_next = (last_date_obj + pd.DateOffset(months=6)).strftime('%Y-%m')
                    combined.loc[f"{date_next} (Est.)"] = [next_val, current_price]
        
        return combined
    except: return None

# (기존 fetch_per_data, fetch_eps_data, fetch_etf_data 함수는 유지)
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
                q1_q, q1_year = (q+1, year) if q < 4 else (1, year+1)
                label_q1 = f"{q1_year}-Q{q1_q}"
                eps_df.loc[label_q1, ticker] = est.loc['0q', 'avg']
                eps_df.loc[label_q1, 'type'] = 'Estimate'
                if predict_mode == "다음 분기 예측":
                    q2_q, q2_year = (q1_q+1, q1_year) if q1_q < 4 else (1, q1_year+1)
                    label_q2 = f"{q2_year}-Q{q2_q}"
                    eps_df.loc[label_q2, ticker] = est.loc['+1q', 'avg']
                    eps_df.loc[label_q2, 'type'] = 'Estimate'
        return eps_df.sort_index()
    except: return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_etf_data(selected_tickers):
    combined_df = pd.DataFrame()
    for ticker in selected_tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start="2016-10-01", interval="1mo", auto_adjust=True)
            if df.empty: continue
            temp_df = df[['Close']].copy()
            temp_df.index = temp_df.index.strftime('%Y-%m')
            temp_df = temp_df[~temp_df.index.duplicated(keep='first')]
            temp_df.columns = [ticker]
            combined_df = temp_df if combined_df.empty else combined_df.join(temp_df, how='outer')
        except: continue
    return combined_df

# --- [UI 레이아웃] ---

with st.sidebar:
    st.title("📂 분석 메뉴")
    main_menu = st.radio(
        "분석 종류를 선택하세요:",
        ("개별종목 적정주가 분석", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 ---
if main_menu == "개별종목 적정주가 분석":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            val_ticker = st.text_input("🏢 분석 티커", "TSLA").upper().strip()
        with col2:
            # 수정 1: 미래 예측 옵션을 라디오 버튼으로 변경 (Default: None)
            val_predict_mode = st.radio(
                "🔮 미래 예측 옵션",
                ("None", "현재 분기 예측", "다음 분기 예측"),
                horizontal=True, index=0
            )
        
        run_val = st.button("적정주가 분석 실행", type="primary", use_container_width=True)

    if run_val and val_ticker:
        combined = fetch_valuation_data(val_ticker, val_predict_mode)
        if combined is not None:
            summary_data = []
            final_price = combined['Close'].iloc[-1]
            target_date = combined.index[-1]
            
            st.subheader(f"📊 {val_ticker} 연도별 시뮬레이션")
            
            for base_year in range(2017, 2026):
                df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                if len(df_plot) < 2: continue
                base_eps = df_plot.iloc[0]['EPS']
                base_price = df_plot.iloc[0]['Close']
                if base_eps <= 0: continue

                scale_factor = base_price / base_eps
                df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                final_fair_value = df_plot.iloc[-1]['Fair_Value']
                gap_pct = ((final_price - final_fair_value) / final_fair_value) * 100
                status = "Overvalued" if gap_pct > 0 else "Undervalued"
                
                summary_data.append({
                    "Base Year": base_year,
                    "Multiplier (PER)": f"{scale_factor:.1f}x",
                    "Fair Value": f"${final_fair_value:.2f}",
                    "Current Price": f"${final_price:.2f}",
                    "Gap (%)": f"{gap_pct:+.2f}%",
                    "Status": status
                })

                # 수정 2: 그래프 크기 80% 축소 (7.7 x 3.2)
                fig, ax = plt.subplots(figsize=(7.7, 3.2), facecolor='white')
                
                # 수정 3: 좌측 상단 커스텀 범례 텍스트 추가
                ax.text(0.02, 0.92, "● Price", color='#1f77b4', transform=ax.transAxes, fontweight='bold', fontsize=9)
                ax.text(0.12, 0.92, "■ EPS", color='#d62728', transform=ax.transAxes, fontweight='bold', fontsize=9)

                ax.plot(df_plot.index, df_plot['Close'], label='Market Price', color='#1f77b4', linewidth=2.0, marker='o', markersize=4)
                ax.plot(df_plot.index, df_plot['Fair_Value'], label=f'Fair Value (Base: {base_year})', color='#d62728', linestyle='--', marker='s', markersize=4)

                for i, idx in enumerate(df_plot.index):
                    if "(Est.)" in idx:
                        ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.15)
                
                apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            if summary_data:
                st.divider()
                st.subheader(f"📋 Valuation Summary (Target: {target_date})")
                
                # 수정 4: 테이블 가로 길이 1/2 수준으로 축소
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(
                    summary_df, 
                    use_container_width=False, 
                    width=600, # 가로 너비 제한
                    hide_index=True
                )
                st.info("💡 'Undervalued'가 많은 종목일수록 역사적 밸류에이션 하단에 있을 가능성이 높습니다.")
        else:
            st.error("데이터를 가져올 수 없습니다. 티커를 확인해 주세요.")

# --- 메뉴 2: 기업 가치 비교 (기존 코드 유지) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            ticker_input = st.text_input("🏢 티커 입력", "AAPL, MSFT, NVDA")
        with col2:
            start_year = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3:
            predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        
        selected_metric = st.radio("📈 분석 지표 선택", ("PER 증감률 (%)", "EPS 성장률 (%)"), horizontal=True)
        analyze_btn = st.button("데이터 분석 실행", type="primary", use_container_width=True)

    if analyze_btn:
        tickers = [t.strip().upper() for t in ticker_input.replace(',', ' ').split() if t.strip()]
        if selected_metric == "PER 증감률 (%)":
            master_per = pd.DataFrame()
            for t in tickers:
                s = fetch_per_data(t, predict_mode)
                if s is not None: master_per[t] = s
            if not master_per.empty:
                master_per = master_per[master_per.index >= f"{start_year}-01-01"].sort_index()
                indexed_per = (master_per / master_per.iloc[0] - 1) * 100
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
                x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_per.index]
                for i, ticker in enumerate(indexed_per.columns):
                    series = indexed_per[ticker].dropna()
                    f_count = 1 if predict_mode == "현재 분기 예측" else (2 if predict_mode == "다음 분기 예측" else 0)
                    h_end = len(series) - f_count
                    ax.plot(range(h_end), series.values[:h_end], marker='o', label=f"{ticker} ({series.values[-1]:+.1f}%)", color=colors[i], linewidth=2.5)
                    if f_count > 0:
                        ax.plot(range(h_end-1, len(series)), series.values[h_end-1:], linestyle='--', color=colors[i], linewidth=2.0, alpha=0.8)
                apply_strong_style(ax, f"Relative PER Change since {start_year}", "Change (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_xticks(range(len(indexed_per))); ax.set_xticklabels(x_labels, rotation=45)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
                st.pyplot(fig)
        else: # EPS
            all_eps = []
            for t in tickers:
                df = fetch_eps_data(t, predict_mode)
                if not df.empty: all_eps.append(df)
            if all_eps:
                full_idx = sorted(list(set().union(*(d.index for d in all_eps))))
                filtered_idx = [idx for idx in full_idx if idx >= f"{start_year}-Q1"]
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                for i, df in enumerate(all_eps):
                    t = [c for c in df.columns if c != 'type'][0]
                    plot_df = df.reindex(filtered_idx)
                    valid_data = plot_df[plot_df[t].notna()]
                    if valid_data.empty: continue
                    norm_vals = (plot_df[t] / valid_data[t].iloc[0] - 1) * 100
                    color = plt.cm.Set1(i % 9)
                    act_mask = plot_df['type'] == 'Actual'
                    last_act = np.where(act_mask)[0][-1] if any(act_mask) else 0
                    ax.plot(range(last_act + 1), norm_vals.iloc[:last_act + 1], marker='o', label=f"{t} ({norm_vals.dropna().values[-1]:+.1f}%)", color=color, linewidth=2.5)
                    if predict_mode != "None":
                        ax.plot(range(last_act, len(filtered_idx)), norm_vals.iloc[last_act:], linestyle='--', color=color, linewidth=2.0)
                apply_strong_style(ax, f"Normalized EPS Growth since {start_year}-Q1", "Growth (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_xticks(range(len(filtered_idx))); ax.set_xticklabels(filtered_idx, rotation=45)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
                st.pyplot(fig)

# --- 메뉴 3: ETF 섹터 수익률 (기존 코드 유지) ---
else:
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sector_list = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
            selected_etfs = st.multiselect("🌐 ETF 선택", sector_list, default=["SPY", "QQQ", "XLK", "XLE"])
        with col2:
            start_year_etf = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3:
            start_q_etf = st.selectbox("🔢 기준 분기", [1, 2, 3, 4], index=0)
        run_etf_btn = st.button("ETF 수익률 분석 시작", type="primary", use_container_width=True)

    if run_etf_btn and selected_etfs:
        df_etf = fetch_etf_data(selected_etfs)
        start_date = f"{start_year_etf}-{str((start_q_etf-1)*3 + 1).zfill(2)}"
        if any(df_etf.index >= start_date):
            valid_start = df_etf.index[df_etf.index >= start_date][0]
            norm_etf = (df_etf.loc[valid_start:] / df_etf.loc[valid_start:].iloc[0] - 1) * 100
            last_vals = norm_etf.iloc[-1].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
            vivid_colors = plt.cm.get_cmap('tab10', len(selected_etfs))
            for i, ticker in enumerate(last_vals.index):
                lw = 4.0 if ticker in ["SPY", "QQQ"] else 2.5
                ax.plot(norm_etf.index, norm_etf[ticker], label=f"{ticker} ({last_vals[ticker]:+.1f}%)", color=vivid_colors(i), linewidth=lw)
            apply_strong_style(ax, f"ETF Performance since {valid_start}", "Return (%)")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ticks = [d for d in norm_etf.index if d.endswith(('-01', '-04', '-07', '-10'))]
            ax.set_xticks(ticks); ax.set_xticklabels(ticks, rotation=45)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
            st.pyplot(fig)
