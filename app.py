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
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "개별종목 적정주가 분석 4", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 1 ---
if main_menu == "개별종목 적정주가 분석 1":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        val_ticker = col1.text_input("🏢 분석 티커", "TSLA").upper().strip()
        val_predict_mode = col2.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_val = st.button("적정주가 분석 실행", type="primary", use_container_width=True)
    if run_val and val_ticker:
        combined = fetch_valuation_data(val_ticker, val_predict_mode)
        if combined is not None:
            final_price = combined['Close'].iloc[-1]
            for base_year in range(2017, 2026):
                df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                if len(df_plot) < 2 or df_plot.iloc[0]['EPS'] <= 0: continue
                scale_factor = df_plot.iloc[0]['Close'] / df_plot.iloc[0]['EPS']
                df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                gap_pct = ((final_price - df_plot.iloc[-1]['Fair_Value']) / df_plot.iloc[-1]['Fair_Value']) * 100
                fig, ax = plt.subplots(figsize=(7.7, 3.2), facecolor='white')
                ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', linewidth=2.0, marker='o', markersize=4)
                ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', marker='s', markersize=4)
                apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig)

# --- 메뉴 2: 개별종목 적정주가 분석 2 ---
elif main_menu == "개별종목 적정주가 분석 2":
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            v2_ticker = st.text_input("🏢 분석 티커 입력", "PAYX").upper().strip()
        with col2:
            st.write("")
            st.write("")
            run_v2 = st.button("정밀 가치 분석 실행", type="primary", use_container_width=True)

    if run_v2 and v2_ticker:
        try:
            with st.spinner('데이터를 수집하고 분석 중입니다...'):
                stock = yf.Ticker(v2_ticker)
                
                # 1. 과거 실적 수집
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
                        if raw_eps.index.tz is not None:
                            raw_eps.index = raw_eps.index.tz_localize(None)
                        break

                raw_eps = raw_eps[raw_eps.index >= "2017-01-01"]
                
                # 2. 주가 및 예측치 수집
                price_history = stock.history(start="2017-01-01", interval="1d")
                price_df = price_history['Close'].copy()
                if price_df.index.tz is not None:
                    price_df.index = price_df.index.tz_localize(None)
                    
                current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
                estimates = stock.earnings_estimate
                current_q_est = estimates['avg'].iloc[0] if estimates is not None else 0

                # 3. 타겟 EPS 계산
                recent_3_actuals = raw_eps['EPS'].iloc[-3:].sum()
                final_target_eps = recent_3_actuals + current_q_est

                # 4. 4분기 단위 프로세싱
                processed_data = []
                for i in range(0, len(raw_eps) - 3, 4):
                    group = raw_eps.iloc[i:i+4]
                    eps_sum = group['EPS'].sum()
                    start_date, end_date = group.index[0], group.index[-1]
                    avg_price = price_df[start_date:end_date].mean()
                    is_last_row = (i + 4 >= len(raw_eps))
                    
                    eps_display = f"{eps_sum:.2f}"
                    if is_last_row:
                        eps_display = f"{final_target_eps:.2f}(예상)"
                        eps_sum = final_target_eps
                    
                    processed_data.append({
                        '기준 연도': f"{start_date.year}년",
                        '4분기 EPS합': eps_display,
                        '평균 주가': f"${avg_price:.2f}",
                        '평균 PER': avg_price / eps_sum if eps_sum > 0 else 0,
                        'EPS_Val': eps_sum
                    })

                # 5. UI 출력
                st.subheader(f"🔍 [{v2_ticker}] 발표일 기준 과거 밸류에이션 기록")
                st.markdown(f"**분석 기준 EPS:** `${final_target_eps:.2f}` (최근 3개 확정 + 1개 예측)")
                
                display_list = []
                past_pers = [d['평균 PER'] for d in processed_data if d['평균 PER'] > 0]
                avg_past_per = np.mean(past_pers) if past_pers else 0

                for data in processed_data:
                    fair_price = final_target_eps * data['평균 PER']
                    diff_pct = ((current_price / fair_price) - 1) * 100
                    status = "🔴 고평가" if current_price > fair_price else "🔵 저평가"
                    
                    display_list.append({
                        "기준 연도": data['기준 연도'],
                        "4분기 EPS합": data['4분기 EPS합'],
                        "평균 주가": data['평균 주가'],
                        "평균 PER": f"{data['평균 PER']:.1f}x",
                        "적정주가 가치": f"${fair_price:.2f}",
                        "현재가 판단": f"{abs(diff_pct):.1f}% {status}"
                    })

                st.dataframe(
                    pd.DataFrame(display_list),
                    use_container_width=False,
                    width=750,
                    hide_index=True
                )

                # 요약 정보
                current_fair_value = final_target_eps * avg_past_per
                current_diff = ((current_price / current_fair_value) - 1) * 100
                c_status = "고평가" if current_price > current_fair_value else "저평가"
                
                st.success(f"""
                **[최종 요약]**
                * 현재 실시간 주가: **${current_price:.2f}**
                * 과거 평균 PER(**{avg_past_per:.1f}x**) 기준 적정가: **${current_fair_value:.2f}**
                * 결과: 현재 주가는 적정가 대비 **{abs(current_diff):.1f}% {c_status}** 상태입니다.
                """)
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")




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
            url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
            headers = {'User-Agent': 'Mozilla/5.0'}
            dfs = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))
            target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('PER|EPS').any()), None)
            if target_df is not None:
                per_raw = target_df[target_df.index.astype(str).str.contains('PER')].transpose()
                per_series = pd.to_numeric(per_raw.iloc[:, 0], errors='coerce').dropna()
                per_series.index = pd.to_datetime(per_series.index, format='%y.%m.%d')
                per_series = per_series[per_series.index >= f"{base_year}-01-01"]
                fig, ax = plt.subplots(figsize=(8.0, 4.0), facecolor='white')
                ax.plot(per_series.index.strftime('%y.%m'), per_series.values, marker='o', color='#34495e', linewidth=2, label='Forward PER')
                ax.axhline(per_series.mean(), color='#e74c3c', linestyle='--', label=f'Mean: {per_series.mean():.1f}')
                apply_strong_style(ax, f"{v3_ticker} PER Valuation Trend", "PER Ratio")
                plt.xticks(rotation=45)
                ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black')
                st.pyplot(fig)
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 4: 개별종목 적정주가 분석 4 (테이블 너비 20% 확대: 550) ---
elif main_menu == "개별종목 적정주가 분석 4":
    with st.container(border=True):
        v4_ticker = st.text_input("🏢 분석 티커 입력 (PEG 분석)", "AAPL").upper().strip()
        run_v4 = st.button("연도별 정밀 PEG 분석 실행", type="primary", use_container_width=True)

    if run_v4 and v4_ticker:
        try:
            with st.spinner(f"[{v4_ticker}] 연도별 정밀 PEG 분석 중..."):
                url = f"https://www.choicestock.co.kr/search/invest/{v4_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                dfs = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))
                target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS').any()), None)
                if target_df is not None:
                    target_df = target_df.set_index(target_df.columns[0])
                    eps_df = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                    eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
                    eps_df = eps_df.dropna().sort_index()
                    eps_df.columns = ['Quarterly_EPS']
                    stock = yf.Ticker(v4_ticker)
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                    estimates = stock.earnings_estimate
                    latest_date = eps_df.index[-1]
                    def get_ttm(idx): return eps_df['Quarterly_EPS'].iloc[idx-3 : idx+1].sum() if idx >= 3 else None

                    def display_peg_table(title, date, data_list):
                        st.subheader(f"📌 {title} (기준일: {date.date()})")
                        df_res = pd.DataFrame(data_list)
                        df_res.columns = ['분석 기간', '과거 TTM EPS', '기준 TTM EPS', '연평균성장률(%)', 'PER', 'PEG']
                        # 너비를 기존 450에서 약 20% 늘린 550으로 설정
                        st.dataframe(df_res.style.format({
                            '과거 TTM EPS': '{:.2f}', '기준 TTM EPS': '{:.2f}',
                            '연평균성장률(%)': '{:.2f}', 'PER': '{:.2f}', 'PEG': '{:.2f}'
                        }), width=550, hide_index=True)

                    results = []
                    per_val = current_price / get_ttm(len(eps_df)-1)
                    for y in range(5, 0, -1):
                        t_idx = len(eps_df)-1 - (y*4)
                        if t_idx >= 3:
                            past_eps, curr_eps = get_ttm(t_idx), get_ttm(len(eps_df)-1)
                            growth = ((curr_eps/past_eps)**(1/y)-1)*100
                            results.append({
                                'period': f"최근 {y}년 연간", 'past': past_eps, 'curr': curr_eps,
                                'growth': growth, 'per': per_val, 'peg': per_val/growth if growth > 0 else 0
                            })
                    display_peg_table("[확정 실적 기준] 연간 PEG", latest_date, results)
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 5: 기업 가치 비교 (PER/EPS) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        ticker_input = col1.text_input("🏢 티커 입력", "AAPL, MSFT, NVDA")
        start_year = col2.number_input("📅 기준 연도", 2010, 2025, 2020)
        predict_mode = col3.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True)
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
                for ticker in indexed_per.columns:
                    ax.plot(indexed_per.index.strftime('%yQ%q'), indexed_per[ticker], marker='o', label=ticker, linewidth=2)
                apply_strong_style(ax, f"Relative PER Change since {start_year}", "Change (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                plt.xticks(rotation=45); ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                st.pyplot(fig)
        else:
            all_eps = []
            for t in tickers:
                df = fetch_eps_data(t, predict_mode)
                if not df.empty: all_eps.append(df)
            if all_eps:
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                for df in all_eps:
                    t = [c for c in df.columns if c != 'type'][0]
                    plot_df = df[df.index >= f"{start_year}-Q1"]
                    norm_vals = (plot_df[t] / plot_df[t].iloc[0] - 1) * 100
                    ax.plot(plot_df.index, norm_vals, marker='o', label=t, linewidth=2)
                apply_strong_style(ax, f"Normalized EPS Growth since {start_year}-Q1", "Growth (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                plt.xticks(rotation=45); ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                st.pyplot(fig)

# --- 메뉴 6: ETF 섹터 수익률 분석 ---
else:
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        selected_etfs = col1.multiselect("🌐 ETF 선택", ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"], default=["SPY", "QQQ", "XLK", "XLE"])
        start_year_etf = col2.number_input("📅 기준 연도", 2010, 2025, 2020)
        start_q_etf = col3.selectbox("🔢 기준 분기", [1, 2, 3, 4])
        run_etf_btn = st.button("ETF 수익률 분석 시작", type="primary", use_container_width=True)
    if run_etf_btn and selected_etfs:
        df_etf = fetch_etf_data(selected_etfs)
        start_date = f"{start_year_etf}-{str((start_q_etf-1)*3 + 1).zfill(2)}"
        norm_etf = (df_etf.loc[start_date:] / df_etf.loc[start_date:].iloc[0] - 1) * 100
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        for ticker in norm_etf.columns:
            ax.plot(norm_etf.index, norm_etf[ticker], label=ticker, linewidth=2.5 if ticker in ["SPY", "QQQ"] else 1.5)
        apply_strong_style(ax, f"ETF Performance since {start_date}", "Return (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xticks(rotation=45); ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        st.pyplot(fig)

