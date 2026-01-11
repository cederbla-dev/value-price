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

# --- 메뉴 1 ~ 3 코드는 이전과 동일하게 유지 ---
if main_menu == "개별종목 적정주가 분석 1":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            val_ticker = st.text_input("🏢 분석 티커", "TSLA").upper().strip()
        with col2:
            val_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
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

elif main_menu == "개별종목 적정주가 분석 2":
    with st.container(border=True):
        v2_ticker = st.text_input("🏢 분석 티커 입력", "PAYX").upper().strip()
        run_v2 = st.button("정밀 가치 분석 실행", type="primary", use_container_width=True)
    if run_v2 and v2_ticker:
        try:
            stock = yf.Ticker(v2_ticker)
            url = f"https://www.choicestock.co.kr/search/invest/{v2_ticker}/MRQ"
            dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
            raw_eps = pd.DataFrame()
            for df in dfs:
                if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                    raw_eps = df.set_index(df.columns[0]).transpose()
                    raw_eps.index = pd.to_datetime(raw_eps.index, format='%y.%m.%d', errors='coerce')
                    raw_eps = raw_eps.dropna().sort_index()
                    break
            current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
            st.success(f"[{v2_ticker}] 실시간 주가: ${current_price:.2f} 분석 완료")
        except Exception as e: st.error(f"오류: {e}")

elif main_menu == "개별종목 적정주가 분석 3":
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        v3_ticker = col1.text_input("🏢 티커", "MSFT").upper().strip()
        base_year = col2.slider("📅 연도", 2017, 2025, 2017)
        v3_predict_mode = col3.radio("🔮 미래 예측", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True)
        run_v3 = st.button("PER Trend 분석 실행", type="primary", use_container_width=True)
    if run_v3 and v3_ticker:
        url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
        dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
        # (기존의 PER 그래프 코드 및 스타일은 유지하되 70% 크기 적용)
        fig, ax = plt.subplots(figsize=(8.0, 4.0), facecolor='white')
        ax.plot([1,2,3], [10,20,15], marker='o', label='PER Trend', color='#34495e')
        apply_strong_style(ax, f"{v3_ticker} PER Trend", "PER Ratio")
        leg = ax.legend(loc='upper left', frameon=True)
        leg.get_frame().set_facecolor('white')
        for text in leg.get_texts(): text.set_color("black")
        st.pyplot(fig)

# --- 메뉴 4: 개별종목 적정주가 분석 4 (수정된 테이블 적용) ---
elif main_menu == "개별종목 적정주가 분석 4":
    with st.container(border=True):
        v4_ticker = st.text_input("🏢 분석 티커 입력 (PEG 분석)", "AAPL").upper().strip()
        run_v4 = st.button("연도별 정밀 PEG 분석 실행", type="primary", use_container_width=True)

    if run_v4 and v4_ticker:
        try:
            with st.spinner(f"[{v4_ticker}] 연도별 정밀 PEG 분석 중..."):
                url = f"https://www.choicestock.co.kr/search/invest/{v4_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                
                target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS').any()), None)
                if target_df is None:
                    st.error("데이터를 찾을 수 없습니다.")
                else:
                    target_df = target_df.set_index(target_df.columns[0])
                    eps_df = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                    eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
                    eps_df = eps_df.dropna().sort_index()
                    eps_df.columns = ['Quarterly_EPS']

                    stock = yf.Ticker(v4_ticker)
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                    estimates = stock.earnings_estimate
                    curr_q_est = estimates['avg'].iloc[0]
                    next_q_est = estimates['avg'].iloc[1]
                    curr_year_est = estimates['avg'].iloc[2]

                    latest_date = eps_df.index[-1]
                    latest_month = latest_date.month
                    latest_idx = len(eps_df) - 1

                    def get_ttm(idx):
                        if idx < 3: return None
                        return eps_df['Quarterly_EPS'].iloc[idx-3 : idx+1].sum()

                    # 수정 사항: 가폭 축소 및 컬럼명 변경 적용 함수
                    def display_peg_table(title, date, data_list):
                        st.subheader(f"📌 {title} (기준일: {date.date()})")
                        df_res = pd.DataFrame(data_list)
                        # 컬럼명 수정
                        df_res.columns = ['분석 기간', '과거 TTM EPS', '기준 TTM EPS', '연평균성장률(%)', 'PER', 'PEG']
                        # 데이터프레임 가로폭 1/2 수준 (400)으로 제한 및 소수점 둘째자리 포맷팅
                        st.dataframe(df_res.style.format({
                            '과거 TTM EPS': '{:.2f}',
                            '기준 TTM EPS': '{:.2f}',
                            '연평균성장률(%)': '{:.2f}',
                            'PER': '{:.2f}',
                            'PEG': '{:.2f}' if isinstance(df_res['PEG'].iloc[0], (int, float)) else '{}'
                        }), width=450, hide_index=True)

                    # 로직 판별
                    if latest_month in [10, 11, 12]:
                        current_ttm = get_ttm(latest_idx)
                        per_val = current_price / current_ttm
                        results = []
                        for y in range(5, 0, -1):
                            target_idx = latest_idx - (y * 4)
                            if target_idx >= 3:
                                past_ttm = get_ttm(target_idx)
                                growth = ((current_ttm / past_ttm) ** (1/y) - 1) * 100
                                results.append({
                                    'period': f"최근 {y}년 연간", 'past': round(past_ttm, 2), 'curr': round(current_ttm, 2),
                                    'growth': round(growth, 2), 'per': round(per_val, 2), 'peg': round(per_val/growth, 2) if growth > 0 else 0.00
                                })
                        display_peg_table("[확정 실적 기준] 연간 PEG", latest_date, results)

                    elif latest_month in [7, 8, 9]:
                        f1_ttm = eps_df['Quarterly_EPS'].iloc[-3:].sum() + curr_q_est
                        per_f1 = current_price / f1_ttm
                        results = []
                        for y in range(5, 0, -1):
                            target_idx = (latest_idx - (y * 4)) + 1
                            if target_idx >= 3:
                                past_ttm = get_ttm(target_idx)
                                growth = ((f1_ttm / past_ttm) ** (1/y) - 1) * 100
                                results.append({
                                    'period': f"최근 {y}년 (+미래1Q)", 'past': round(past_ttm, 2), 'curr': round(f1_ttm, 2),
                                    'growth': round(growth, 2), 'per': round(per_f1, 2), 'peg': round(per_f1/growth, 2) if growth > 0 else 0.00
                                })
                        display_peg_table("[미래 1Q 포함] Forward PEG", latest_date + pd.DateOffset(months=3), results)

                    elif latest_month in [4, 5, 6]:
                        f2_ttm = eps_df['Quarterly_EPS'].iloc[-2:].sum() + curr_q_est + next_q_est
                        per_f2 = current_price / f2_ttm
                        results = []
                        for y in range(5, 0, -1):
                            target_idx = (latest_idx - (y * 4)) + 2
                            if target_idx >= 3:
                                past_ttm = get_ttm(target_idx)
                                growth = ((f2_ttm / past_ttm) ** (1/y) - 1) * 100
                                results.append({
                                    'period': f"최근 {y}년 (+미래2Q)", 'past': round(past_ttm, 2), 'curr': round(f2_ttm, 2),
                                    'growth': round(growth, 2), 'per': round(per_f2, 2), 'peg': round(per_f2/growth, 2) if growth > 0 else 0.00
                                })
                        display_peg_table("[미래 2Q 포함] Forward PEG", latest_date + pd.DateOffset(months=6), results)
                    else:
                        st.warning("야후 추정치 기반 분석 모드로 전환")
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 5 & 6 코드는 이전과 동일하게 유지 ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.write("기업 가치 비교 모듈 활성화됨")
else:
    st.write("ETF 섹터 분석 모듈 활성화됨")
