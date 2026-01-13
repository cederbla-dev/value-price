import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import datetime, timedelta
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
        stock = yf.Ticker(ticker)
        price_df = stock.history(start="2017-01-01", interval="1mo", auto_adjust=False)
        if price_df.index.tz is not None: price_df.index = price_df.index.tz_localize(None)
        price_df.index = price_df.index.strftime('%Y-%m')
        price_df = price_df[['Close']].copy()
        price_df = price_df[~price_df.index.duplicated(keep='last')]
        combined = pd.merge(eps_df, price_df, left_index=True, right_index=True, how='inner')
        combined = combined.sort_index(ascending=True)
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
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "개별종목 적정주가 분석 4", "개별종목 적정주가 분석 5", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 1 ---
if main_menu == "개별종목 적정주가 분석 1":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        val_ticker = col1.text_input("🏢 분석 티커 입력", "TSLA").upper().strip()
        val_predict_mode = col2.radio(
            "🔮 미래 예측 옵션 (Estimates)", 
            ("None", "현재 분기 예측", "다음 분기 예측"), 
            horizontal=True, 
            index=0
        )
        run_val = st.button("적정주가 분석 실행", type="primary", use_container_width=True)

    if run_val and val_ticker:
        with st.spinner(f"[{val_ticker}] 데이터를 정밀 분석 중입니다..."):
            combined = fetch_valuation_data(val_ticker, val_predict_mode)
            if combined is not None and not combined.empty:
                final_price = combined['Close'].iloc[-1]
                target_date_label = combined.index[-1]
                summary_list = []
                st.subheader(f"📈 {val_ticker} 연도별 적정주가 시뮬레이션")
                for base_year in range(2017, 2026):
                    df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                    if len(df_plot) < 2 or df_plot.iloc[0]['EPS'] <= 0: continue
                    scale_factor = df_plot.iloc[0]['Close'] / df_plot.iloc[0]['EPS']
                    df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                    last_fair_value = df_plot.iloc[-1]['Fair_Value']
                    gap_pct = ((final_price - last_fair_value) / last_fair_value) * 100
                    status = "🔴 고평가" if gap_pct > 0 else "🔵 저평가"
                    summary_list.append({
                        "기준 연도": f"{base_year}년",
                        "기준 PER": f"{scale_factor:.1f}x",
                        "적정 주가": f"${last_fair_value:.2f}",
                        "현재 주가": f"${final_price:.2f}",
                        "괴리율 (%)": f"{gap_pct:+.1f}%",
                        "상태": status
                    })
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', linewidth=2.0, marker='o', markersize=4, label='Price')
                    ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', marker='s', markersize=4, label='EPS')
                    for i, idx in enumerate(df_plot.index):
                        if "(Est.)" in str(idx): ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.1)
                    apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                    plt.xticks(rotation=45)
                    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)
                    for text in leg.get_texts():
                        if text.get_text() == 'Price':
                            text.set_color('#1f77b4')
                            text.set_weight('bold')
                        elif text.get_text() == 'EPS':
                            text.set_color('#d62728')
                            text.set_weight('bold')
                    st.pyplot(fig)
                    plt.close(fig)
                if summary_list:
                    st.write("\n")
                    st.markdown("---")
                    st.subheader(f"📊 {val_ticker} 밸류에이션 종합 요약")
                    summary_df = pd.DataFrame(summary_list)
                    main_col, _ = st.columns([6, 4]) 
                    with main_col:
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    st.info(f"💡 **분석 가이드**: 다수의 기준 연도 대비 '저평가'가 많다면 현재 주가는 매력적인 구간일 확률이 높습니다.")
                else: st.warning("분석 가능한 데이터 부족")
            else: st.error("데이터 수집 실패")

# --- 메뉴 2: 개별종목 적정주가 분석 2 ---
elif main_menu == "개별종목 적정주가 분석 2":
    with st.container(border=True):
        col1, col2, col3 = st.columns([0.5, 0.5, 1], vertical_alignment="bottom")
        v2_ticker = col1.text_input("🏢 분석 티커 입력", "AAPL").upper().strip()
        run_v2 = col2.button("당해 EPS 기반 분석", type="primary", use_container_width=True)
    if run_v2 and v2_ticker:
        try:
            with st.spinner('분석 중...'):
                stock = yf.Ticker(v2_ticker)
                url = f"https://www.choicestock.co.kr/search/invest/{v2_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                raw_eps = pd.DataFrame()
                for df in dfs:
                    if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                        target_df = df.set_index(df.columns[0])
                        raw_eps = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                        raw_eps.index = pd.to_datetime(raw_eps.index, format='%y.%m.%d', errors='coerce')
                        raw_eps = raw_eps.dropna().sort_index()
                        raw_eps.columns = ['EPS']
                        break
                price_df = stock.history(start="2017-01-01", interval="1d")['Close']
                if price_df.index.tz is not None: price_df.index = price_df.index.tz_localize(None)
                current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
                estimates = stock.earnings_estimate
                current_q_est = estimates['avg'].iloc[0] if estimates is not None else 0
                final_target_eps = raw_eps['EPS'].iloc[-3:].sum() + current_q_est
                processed_data = []
                for i in range(0, len(raw_eps) - 3, 4):
                    group = raw_eps.iloc[i:i+4]
                    eps_sum = group['EPS'].sum()
                    avg_price = price_df[group.index[0]:group.index[-1]].mean()
                    is_last = (i + 4 >= len(raw_eps))
                    val_eps = final_target_eps if is_last else eps_sum
                    processed_data.append({
                        '기준 연도': f"{group.index[0].year}년",
                        '4분기 EPS합': f"{val_eps:.2f}" + ("(예상)" if is_last else ""),
                        '평균 주가': f"${avg_price:.2f}",
                        '평균 PER': avg_price / val_eps if val_eps > 0 else 0,
                        'EPS_Val': val_eps
                    })
                display_list = []
                past_pers = [d['평균 PER'] for d in processed_data if d['평균 PER'] > 0]
                avg_past_per = np.mean(past_pers) if past_pers else 0
                for data in processed_data:
                    fair_price = final_target_eps * data['평균 PER']
                    status = "🔴 고평가" if current_price > fair_price else "🔵 저평가"
                    display_list.append({
                        "기준 연도": data['기준 연도'], "4분기 EPS합": data['4분기 EPS합'], "평균 주가": data['평균 주가'],
                        "평균 PER": f"{data['평균 PER']:.1f}x", "적정주가 가치": f"${fair_price:.2f}",
                        "현재가 판단": f"{abs(((current_price/fair_price)-1)*100):.1f}% {status}"
                    })
                st.dataframe(pd.DataFrame(display_list), use_container_width=False, width=750, hide_index=True)
                st.success(f"과거 평균 PER({avg_past_per:.1f}x) 기준 적정가: **${final_target_eps * avg_past_per:.2f}**")
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 3: 개별종목 적정주가 분석 3 ---
elif main_menu == "개별종목 적정주가 분석 3":
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        v3_ticker = col1.text_input("🏢 분석 티커", "MSFT").upper().strip()
        base_year = col2.slider("📅 차트 시작 연도", 2017, 2025, 2017)
        v3_predict_mode = col3.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True)
        run_v3 = st.button("PER Trend 분석 실행", type="primary", use_container_width=True)
    if run_v3 and v3_ticker:
        try:
            with st.spinner('분석 중...'):
                url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('PER|EPS').any()), None)
                if target_df is not None:
                    per_raw = target_df[target_df.index.astype(str).str.contains('PER')].transpose()
                    eps_raw = target_df[target_df.index.astype(str).str.contains('EPS')].transpose()
                    combined = pd.DataFrame({
                        'PER': pd.to_numeric(per_raw.iloc[:, 0], errors='coerce'),
                        'EPS': pd.to_numeric(eps_raw.iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
                    }).dropna()
                    combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
                    combined = combined.sort_index()
                    def get_q_label(dt):
                        year = dt.year if dt.day > 5 else (dt - timedelta(days=5)).year
                        q = ((dt.month if dt.day > 5 else (dt - timedelta(days=5)).month)-1)//3 + 1
                        return f"{str(year)[2:]}.Q{q}"
                    combined['Label'] = [get_q_label(d) for d in combined.index]
                    plot_df = combined[combined.index >= f"{base_year}-01-01"].copy()
                    if v3_predict_mode != "None":
                        stock = yf.Ticker(v3_ticker)
                        current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                        est = stock.earnings_estimate
                        if est is not None and not est.empty:
                            hist_eps = combined['EPS'].tolist()
                            plot_df.loc[pd.Timestamp.now()] = [current_price/(sum(hist_eps[-3:]) + est.loc['0q', 'avg']), np.nan, "Next(E)"]
                    avg_per = plot_df['PER'].mean()
                    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                    ax.plot(range(len(plot_df)), plot_df['PER'], marker='o', color='#34495e', label='PER')
                    ax.axhline(avg_per, color='#e74c3c', linestyle='--', label=f'Avg: {avg_per:.1f}')
                    ax.set_xticks(range(len(plot_df)))
                    ax.set_xticklabels(plot_df['Label'], rotation=45)
                    ax.legend()
                    st.pyplot(fig)
                else: st.warning("데이터 부족")
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 4: 개별종목 적정주가 분석 4 ---
elif main_menu == "개별종목 적정주가 분석 4":
    with st.container(border=True):
        v4_ticker = st.text_input("🏢 분석 티커 입력 (PEG 분석)", "AAPL").upper().strip()
        run_v4 = st.button("연도별 정밀 PEG 분석 실행", type="primary", use_container_width=True)
    if run_v4 and v4_ticker:
        try:
            with st.spinner(f"[{v4_ticker}] PEG 분석 중..."):
                url = f"https://www.choicestock.co.kr/search/invest/{v4_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                resp = requests.get(url, headers=headers, timeout=10)
                dfs = pd.read_html(io.StringIO(resp.text))
                target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS', na=False).any()), None)
                if target_df is not None:
                    target_df = target_df.set_index(target_df.columns[0])
                    eps_df = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                    eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
                    eps_df = eps_df.dropna().sort_index()
                    eps_df.columns = ['Quarterly_EPS']
                    stock = yf.Ticker(v4_ticker)
                    current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                    latest_idx = len(eps_df) - 1
                    current_ttm = eps_df['Quarterly_EPS'].iloc[latest_idx-3 : latest_idx+1].sum()
                    per_val = current_price / current_ttm
                    results = []
                    for y in range(5, 0, -1):
                        target_idx = latest_idx - (y * 4)
                        if target_idx >= 3:
                            past_ttm = eps_df['Quarterly_EPS'].iloc[target_idx-3 : target_idx+1].sum()
                            if past_ttm > 0:
                                growth = ((current_ttm / past_ttm) ** (1/y) - 1) * 100
                                results.append({'기간': f"최근 {y}년", '성장률': f"{growth:.1f}%", 'PER': f"{per_val:.1f}x", 'PEG': f"{per_val/growth:.2f}" if growth > 0 else "N/A"})
                    st.table(pd.DataFrame(results))
                else: st.error("데이터 수집 실패")
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 5: 개별종목 적정주가 분석 5 (추가된 부분) ---
elif main_menu == "개별종목 적정주가 분석 5":
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        v5_ticker = col1.text_input("🏢 분석 티커 입력 (PER Band)", "NVDA").upper().strip()
        v5_period = col2.selectbox("📅 분석 기간", ["3년", "5년", "최대"], index=1)
        run_v5 = st.button("PER Band 적정주가 분석 실행", type="primary", use_container_width=True)

    if run_v5 and v5_ticker:
        try:
            with st.spinner("역사적 PER 밴드 데이터를 산출 중입니다..."):
                stock = yf.Ticker(v5_ticker)
                period_map = {"3년": "3y", "5년": "5y", "최대": "max"}
                hist = stock.history(period=period_map[v5_period])
                if hist.empty:
                    st.error("주가 데이터를 가져올 수 없습니다.")
                else:
                    info = stock.info
                    ttm_eps = info.get('trailingEps', 0)
                    if ttm_eps <= 0:
                        st.warning("TTM EPS가 0 이하이므로 분석이 불가능합니다.")
                    else:
                        # PER 히스토리 계산
                        url = f"https://www.choicestock.co.kr/search/invest/{v5_ticker}/MRQ"
                        headers = {'User-Agent': 'Mozilla/5.0'}
                        resp = requests.get(url, headers=headers)
                        dfs = pd.read_html(io.StringIO(resp.text))
                        target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('PER').any()), None)
                        
                        per_series = pd.to_numeric(target_df[target_df.index.str.contains('PER')].transpose().iloc[:, 0], errors='coerce').dropna()
                        
                        min_per, avg_per, max_per = per_series.min(), per_series.mean(), per_series.max()
                        curr_price = hist['Close'].iloc[-1]
                        
                        # 시각화
                        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                        ax.plot(hist.index, hist['Close'], color='black', linewidth=1.5, label='Actual Price')
                        ax.plot(hist.index, [ttm_eps * max_per]*len(hist), '--', color='red', alpha=0.6, label=f'Upper Band ({max_per:.1f}x)')
                        ax.plot(hist.index, [ttm_eps * avg_per]*len(hist), '--', color='green', alpha=0.6, label=f'Avg Band ({avg_per:.1f}x)')
                        ax.plot(hist.index, [ttm_eps * min_per]*len(hist), '--', color='blue', alpha=0.6, label=f'Lower Band ({min_per:.1f}x)')
                        
                        ax.set_title(f"[{v5_ticker}] Historical PER Band Analysis", fontsize=14, fontweight='bold')
                        ax.legend(loc='best')
                        st.pyplot(fig)
                        
                        # 요약 결과
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("하단 적정가 (Min)", f"${ttm_eps*min_per:.2f}")
                        col_b.metric("평균 적정가 (Avg)", f"${ttm_eps*avg_per:.2f}")
                        col_c.metric("상단 적정가 (Max)", f"${ttm_eps*max_per:.2f}")
                        
                        curr_per = curr_price / ttm_eps
                        st.info(f"현재 PER: **{curr_per:.1f}x** | 과거 평균 대비 **{((curr_per/avg_per)-1)*100:+.1f}%** 구간입니다.")
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- 메뉴: 기업 가치 비교 (PER/EPS) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.subheader("📊 주요 기업 가치 지표 비교 분석")
    with st.container(border=True):
        comp_tickers = st.text_input("🏢 비교할 티커들을 입력하세요 (쉼표 구분)", "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA").upper().replace(" ", "")
        comp_predict_mode = st.radio("🔮 예측 옵션 선택", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=1)
        run_comp = st.button("비교 분석 실행", type="primary", use_container_width=True)
    if run_comp and comp_tickers:
        ticker_list = comp_tickers.split(",")
        with st.spinner("데이터 통합 분석 중..."):
            per_results = {}
            for t in ticker_list:
                data = fetch_per_data(t, comp_predict_mode)
                if data is not None: per_results[t] = data
            if per_results:
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                for t, s in per_results.items():
                    ax.plot(s.index, s.values, marker='o', label=t, linewidth=2)
                apply_strong_style(ax, "Historical & Forward PER Comparison", "PER Ratio")
                ax.legend(facecolor='white', edgecolor='black')
                st.pyplot(fig)
            else: st.error("데이터 수집 실패")

# --- 메뉴: ETF 섹터 수익률 분석 ---
elif main_menu == "ETF 섹터 수익률 분석":
    sector_etfs = {
        'Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF', 'Energy': 'XLE',
        'Utilities': 'XLU', 'Consumer Staples': 'XLP', 'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI', 'Materials': 'XLB', 'Real Estate': 'XLRE', 'Communication': 'XLC'
    }
    with st.container(border=True):
        selected_labels = st.multiselect("분석할 섹터를 선택하세요:", list(sector_etfs.keys()), default=list(sector_etfs.keys())[:5])
        base_date_etf = st.date_input("기준 날짜 선택:", datetime(2023, 1, 1))
        run_etf = st.button("섹터 수익률 분석", type="primary", use_container_width=True)
    if run_etf and selected_labels:
        sel_tickers = [sector_etfs[l] for l in selected_labels]
        etf_data = fetch_etf_data(sel_tickers)
        if not etf_data.empty:
            start_str = base_date_etf.strftime('%Y-%m')
            filtered_etf = etf_data[etf_data.index >= start_str]
            if not filtered_etf.empty:
                norm_etf = (filtered_etf / filtered_etf.iloc[0] - 1) * 100
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                for col in norm_etf.columns:
                    ax.plot(norm_etf.index, norm_etf[col], label=col, linewidth=2)
                apply_strong_style(ax, "ETF Sector Cumulative Return (%)", "Return (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                plt.xticks(rotation=45)
                ax.legend(loc='upper left', facecolor='white', edgecolor='black')
                st.pyplot(fig)
