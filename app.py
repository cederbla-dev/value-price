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

# --- [공통] 스타일 및 데이터 처리 함수 ---

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
        ticker = ticker.upper().strip()
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
            current_price = stock.fast_info.get('last_price', price_df['Close'].iloc[-1])
            if est is not None and not est.empty:
                last_date_obj = pd.to_datetime(combined.index[-1].split(' ')[0])
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
        ticker = ticker.upper().strip()
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
                ttm_eps_q1 = sum(combined['EPS'].tolist()[-3:]) + est['avg'].iloc[0]
                combined.loc[last_dt + pd.DateOffset(months=3), 'PER'] = current_price / ttm_eps_q1
                if predict_mode == "다음 분기 예측" and len(est) > 1:
                    ttm_eps_q2 = sum(combined['EPS'].tolist()[-2:]) + est['avg'].iloc[0] + est['avg'].iloc[1]
                    combined.loc[last_dt + pd.DateOffset(months=6), 'PER'] = current_price / ttm_eps_q2
        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined['PER']
    except: return None

@st.cache_data(ttl=3600)
def fetch_eps_data(ticker, predict_mode):
    ticker = ticker.upper().strip()
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
                eps_df.loc[label_q1, ticker] = est['avg'].iloc[0]
                eps_df.loc[label_q1, 'type'] = 'Estimate'
                if predict_mode == "다음 분기 예측" and len(est) > 1:
                    q2_q, q2_year = (q1_q+1, q1_year) if q1_q < 4 else (1, q1_year+1)
                    label_q2 = f"{q2_year}-Q{q2_q}"
                    eps_df.loc[label_q2, ticker] = est['avg'].iloc[1]
                    eps_df.loc[label_q2, 'type'] = 'Estimate'
        return eps_df.sort_index()
    except: return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_etf_data(selected_tickers):
    combined_df = pd.DataFrame()
    for ticker in selected_tickers:
        try:
            stock = yf.Ticker(ticker.upper())
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
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 1 ---
if main_menu == "개별종목 적정주가 분석 1":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1: val_ticker = st.text_input("🏢 분석 티커", "TSLA").upper().strip()
        with col2: val_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_val = st.button("적정주가 분석 실행", type="primary", use_container_width=True)
    if run_val and val_ticker:
        combined = fetch_valuation_data(val_ticker, val_predict_mode)
        if combined is not None:
            summary_data = []
            final_price = combined['Close'].iloc[-1]
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
                summary_data.append({"Base Year": base_year, "Multiplier": f"{scale_factor:.1f}x", "Fair Value": f"${final_fair_value:.2f}", "Current Price": f"${final_price:.2f}", "Gap (%)": f"{gap_pct:+.2f}%", "Status": "Overvalued" if gap_pct > 0 else "Undervalued"})
                fig, ax = plt.subplots(figsize=(7.7, 3.2), facecolor='white')
                ax.plot(df_plot.index, df_plot['Close'], label='Market Price', marker='o', markersize=4)
                ax.plot(df_plot.index, df_plot['Fair_Value'], label='Fair Value', linestyle='--', marker='s', markersize=4)
                apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                st.pyplot(fig)
            st.dataframe(pd.DataFrame(summary_data), width=700, hide_index=True)
        else: st.error("데이터를 불러올 수 없습니다.")

# --- 메뉴 2: 개별종목 적정주가 분석 2 ---
elif main_menu == "개별종목 적정주가 분석 2":
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1: v2_ticker = st.text_input("🏢 분석 티커 입력", "PAYX").upper().strip()
        with col2: st.write(""); st.write(""); run_v2 = st.button("정밀 가치 분석 실행", type="primary", use_container_width=True)
    if run_v2 and v2_ticker:
        try:
            stock = yf.Ticker(v2_ticker)
            url = f"https://www.choicestock.co.kr/search/invest/{v2_ticker}/MRQ"
            headers = {'User-Agent': 'Mozilla/5.0'}
            dfs = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))
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
            current_q_est = estimates['avg'].iloc[0] if estimates is not None and not estimates.empty else 0
            final_target_eps = raw_eps['EPS'].iloc[-3:].sum() + current_q_est
            processed_data = []
            for i in range(0, len(raw_eps) - 3, 4):
                group = raw_eps.iloc[i:i+4]
                eps_val = group['EPS'].sum()
                avg_price = price_df[group.index[0]:group.index[-1]].mean()
                if i + 4 >= len(raw_eps): eps_val = final_target_eps
                processed_data.append({'기준 연도': f"{group.index[0].year}년", '4분기 EPS합': f"{eps_val:.2f}", '평균 주가': f"${avg_price:.2f}", '평균 PER': avg_price / eps_val})
            display_list = []
            for d in processed_data:
                fair = final_target_eps * d['평균 PER']
                status = "🔴 고평가" if current_price > fair else "🔵 저평가"
                display_list.append({"기준 연도": d['기준 연도'], "4분기 EPS합": d['4분기 EPS합'], "평균 주가": d['평균 주가'], "평균 PER": f"{d['평균 PER']:.1f}x", "적정주가 가치": f"${fair:.2f}", "현재가 판단": status})
            st.dataframe(pd.DataFrame(display_list), width=750, hide_index=True)
        except: st.error("분석 오류")

# --- 메뉴 3: 개별종목 적정주가 분석 3 (제공해주신 PER Trend 로직 통합) ---
elif main_menu == "개별종목 적정주가 분석 3":
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            v3_ticker = st.text_input("🏢 분석 티커 입력", "MSFT").upper().strip()
        with col2:
            base_year = st.slider("📅 차트 시작 연도", 2017, 2025, 2017)
        with col3:
            v3_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_v3 = st.button("PER Trend 분석 실행", type="primary", use_container_width=True)

    if run_v3 and v3_ticker:
        try:
            with st.spinner('데이터를 분석 중입니다...'):
                # 1. 데이터 수집 (ChoiceStock)
                url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                
                target_df = None
                for df in dfs:
                    if df.iloc[:, 0].astype(str).str.contains('PER|EPS').any():
                        target_df = df.set_index(df.columns[0])
                        break
                
                if target_df is None:
                    st.error("해당 종목의 실적 테이블을 찾을 수 없습니다.")
                else:
                    per_raw = target_df[target_df.index.astype(str).str.contains('PER')].transpose()
                    eps_raw = target_df[target_df.index.astype(str).str.contains('EPS')].transpose()
                    
                    combined = pd.DataFrame({
                        'PER': pd.to_numeric(per_raw.iloc[:, 0], errors='coerce'),
                        'EPS': pd.to_numeric(eps_raw.iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
                    }).dropna()
                    combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
                    combined = combined.sort_index()

                    # 라벨링 함수
                    def get_q_label(dt):
                        year = dt.year if dt.day > 5 else (dt - timedelta(days=5)).year
                        month = dt.month if dt.day > 5 else (dt - timedelta(days=5)).month
                        q = (month-1)//3 + 1
                        return f"{str(year)[2:]}.Q{q}"

                    combined['Label'] = [get_q_label(d) for d in combined.index]
                    plot_df = combined[combined.index >= f"{base_year}-01-01"].copy()

                    # 2. 야후 파이낸스 데이터
                    stock = yf.Ticker(v3_ticker)
                    current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                    est = stock.earnings_estimate

                    # 3. 예측 로직 적용
                    if v3_predict_mode != "None" and est is not None and not est.empty:
                        hist_eps = combined['EPS'].tolist()
                        last_label = plot_df['Label'].iloc[-1]
                        last_yr = int("20" + last_label.split('.')[0])
                        last_q = int(last_label.split('Q')[1])

                        # Current Qtr
                        curr_q_est = est['avg'].iloc[0]
                        t1_q, t1_yr = (last_q + 1, last_yr) if last_q < 4 else (1, last_yr + 1)
                        ttm_eps_1 = sum(hist_eps[-3:]) + curr_q_est
                        plot_df.loc[pd.Timestamp(f"{t1_yr}-{(t1_q-1)*3+1}-01")] = [current_price/ttm_eps_1, np.nan, f"{str(t1_yr)[2:]}.Q{t1_q}(E)"]

                        # Next Qtr
                        if v3_predict_mode == "다음 분기 예측" and len(est) > 1:
                            next_q_est = est['avg'].iloc[1]
                            t2_q, t2_yr = (t1_q + 1, t1_yr) if t1_q < 4 else (1, t1_yr + 1)
                            ttm_eps_2 = sum(hist_eps[-2:]) + curr_q_est + next_q_est
                            plot_df.loc[pd.Timestamp(f"{t2_yr}-{(t2_q-1)*3+1}-01")] = [current_price/ttm_eps_2, np.nan, f"{str(t2_yr)[2:]}.Q{t2_q}(E)"]

                    # 4. 통계 및 시각화
                    per_series = plot_df['PER'].dropna()
                    avg_per = per_series.mean()
                    median_per = per_series.median()

                    st.subheader(f"📈 {v3_ticker} PER Analysis: Mean vs Median")

                    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                    ax.plot(plot_df['Label'], plot_df['PER'], marker='o', color='#34495e', linewidth=2, markersize=7, label='Forward PER Trend')
                    
                    # 평균/중위값 선
                    ax.axhline(avg_per, color='#e74c3c', linestyle='--', label=f'Average: {avg_per:.2f}')
                    ax.axhline(median_per, color='#8e44ad', linestyle='-.', label=f'Median: {median_per:.2f}')
                    
                    # 예측 구간 하이라이트
                    for i, label in enumerate(plot_df['Label']):
                        if "(E)" in label:
                            ax.axvspan(i-0.4, i+0.4, color='orange', alpha=0.15)
                            ax.text(i, plot_df['PER'].iloc[i]+0.2, f"{plot_df['PER'].iloc[i]:.1f}", ha='center', color='#d35400', fontweight='bold')

                    apply_strong_style(ax, f"PER Trend Band (Since {base_year})", "PER Ratio")
                    plt.xticks(rotation=45)
                    ax.legend(loc='upper left')
                    
                    # 정보 박스
                    info_text = f"Price: ${current_price:.2f}\nMean PER: {avg_per:.2f}\nMedian PER: {median_per:.2f}"
                    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    st.pyplot(fig)
                    
                    # 데이터 요약 표
                    st.dataframe(plot_df[['Label', 'PER']].dropna().iloc[::-1], width=400, hide_index=True)
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}. 티커가 정확한지 확인해 주세요.")

# --- 메뉴 4: 기업 가치 비교 (기존 유지) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1: ticker_input = st.text_input("🏢 티커 입력", "AAPL, MSFT, NVDA")
        with col2: start_year = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3: predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
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
                fig, ax = plt.subplots(figsize=(9.6, 4.8))
                for ticker in indexed_per.columns: ax.plot(indexed_per.index, indexed_per[ticker], marker='o', label=ticker)
                apply_strong_style(ax, "Relative PER Change", "Change (%)")
                ax.legend(); st.pyplot(fig)
        else:
            all_eps = []
            for t in tickers:
                df = fetch_eps_data(t, predict_mode)
                if not df.empty: all_eps.append(df)
            if all_eps:
                fig, ax = plt.subplots(figsize=(9.6, 4.8))
                for df in all_eps:
                    t = [c for c in df.columns if c != 'type'][0]
                    norm_vals = (df[t] / df[t].iloc[0] - 1) * 100
                    ax.plot(df.index, norm_vals, marker='o', label=t)
                apply_strong_style(ax, "EPS Growth", "Growth (%)")
                ax.legend(); st.pyplot(fig)

# --- 메뉴 5: ETF 섹터 수익률 분석 (기존 유지) ---
else:
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sector_list = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
            selected_etfs = st.multiselect("🌐 ETF 선택", sector_list, default=["SPY", "QQQ", "XLK", "XLE"])
        with col2: start_year_etf = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3: start_q_etf = st.selectbox("🔢 기준 분기", [1, 2, 3, 4], index=0)
        run_etf_btn = st.button("ETF 수익률 분석 시작", type="primary", use_container_width=True)
    if run_etf_btn and selected_etfs:
        df_etf = fetch_etf_data(selected_etfs)
        start_date = f"{start_year_etf}-{str((start_q_etf-1)*3 + 1).zfill(2)}"
        if any(df_etf.index >= start_date):
            valid_start = df_etf.index[df_etf.index >= start_date][0]
            norm_etf = (df_etf.loc[valid_start:] / df_etf.loc[valid_start:].iloc[0] - 1) * 100
            fig, ax = plt.subplots(figsize=(10, 5))
            for ticker in norm_etf.columns: ax.plot(norm_etf.index, norm_etf[ticker], label=ticker)
            apply_strong_style(ax, "ETF Performance", "Return (%)")
            ax.legend(); st.pyplot(fig)
