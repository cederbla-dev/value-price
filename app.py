import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt
import numpy as np
import warnings
from datetime import timedelta

# 기본 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock & ETF Professional Analyzer", layout="wide")

# --- [공통] 스타일 및 데이터 처리 함수 ---

def apply_strong_style(ax, title, ylabel):
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='black')
    ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', color='black')
    ax.grid(True, linestyle='--', alpha=0.5, color='#d3d3d3')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.2)
    ax.tick_params(axis='both', colors='black', labelsize=8)

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
        return combined
    except: return None

# --- [UI 레이아웃] ---

with st.sidebar:
    st.title("📂 분석 메뉴")
    main_menu = st.radio(
        "분석 종류를 선택하세요:",
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 3: 개별종목 적정주가 분석 3 (수정된 버전) ---
if main_menu == "개별종목 적정주가 분석 3":
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
            with st.spinner('데이터 분석 중...'):
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
                    st.error("실적 데이터를 찾을 수 없습니다. 티커를 확인해 주세요.")
                else:
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
                        month = dt.month if dt.day > 5 else (dt - timedelta(days=5)).month
                        q = (month-1)//3 + 1
                        return f"{str(year)[2:]}.Q{q}"

                    combined['Label'] = [get_q_label(d) for d in combined.index]
                    plot_df = combined[combined.index >= f"{base_year}-01-01"].copy()

                    stock = yf.Ticker(v3_ticker)
                    current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                    est = stock.earnings_estimate

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

                        if v3_predict_mode == "다음 분기 예측" and len(est) > 1:
                            next_q_est = est['avg'].iloc[1]
                            t2_q, t2_yr = (t1_q + 1, t1_yr) if t1_q < 4 else (1, t1_yr + 1)
                            ttm_eps_2 = sum(hist_eps[-2:]) + curr_q_est + next_q_est
                            plot_df.loc[pd.Timestamp(f"{t2_yr}-{(t2_q-1)*3+1}-01")] = [current_price/ttm_eps_2, np.nan, f"{str(t2_yr)[2:]}.Q{t2_q}(E)"]

                    per_series = plot_df['PER'].dropna()
                    avg_per = per_series.mean()
                    median_per = per_series.median()

                    st.subheader(f"📈 {v3_ticker} PER Trend Analysis")

                    # 그래프 크기 축소 (기존 12x6 -> 8.5x4.5 로 약 70% 수준 조정)
                    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor='white')
                    
                    # 메인 트렌드 선
                    ax.plot(plot_df['Label'], plot_df['PER'], marker='o', color='#34495e', linewidth=2, markersize=6, label='Forward PER Trend')
                    
                    # 평균 및 중위값 선 추가 (범례 포함)
                    ax.axhline(avg_per, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Mean (평균값): {avg_per:.2f}')
                    ax.axhline(median_per, color='#8e44ad', linestyle='-.', linewidth=1.5, label=f'Median (중위값): {median_per:.2f}')
                    
                    # 예측 구간 하이라이트
                    for i, label in enumerate(plot_df['Label']):
                        if "(E)" in label:
                            ax.axvspan(i-0.4, i+0.4, color='orange', alpha=0.1)
                    
                    apply_strong_style(ax, f"PER Valuation Trend (Since {base_year})", "PER Ratio")
                    plt.xticks(rotation=45)
                    
                    # 범례 위치 및 설명 보강
                    ax.legend(loc='upper left', fontsize=8, frameon=True, shadow=True)
                    
                    # 우측 하단 정보 박스
                    info_text = f"Current Price: ${current_price:.2f}\nMean: {avg_per:.2f}\nMedian: {median_per:.2f}"
                    ax.text(0.97, 0.05, info_text, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right', 
                            fontsize=8, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#d3d3d3'))
                    
                    st.pyplot(fig)
                    
                    with st.expander("상세 데이터 보기"):
                        st.dataframe(plot_df[['Label', 'PER']].dropna().iloc[::-1], use_container_width=True)
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# --- 나머지 메뉴 (분석 1, 2, 비교, ETF) ---
elif main_menu == "개별종목 적정주가 분석 1":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1: val_ticker = st.text_input("🏢 분석 티커", "TSLA").upper().strip()
        with col2: val_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True)
        run_val = st.button("분석 실행")
    if run_val and val_ticker:
        combined = fetch_valuation_data(val_ticker, val_predict_mode)
        if combined is not None:
            final_price = combined['Close'].iloc[-1]
            for base_year in range(2017, 2026):
                df_p = combined[combined.index >= f'{base_year}-01'].copy()
                if len(df_p) < 2: continue
                scale = df_p.iloc[0]['Close'] / df_p.iloc[0]['EPS'] if df_p.iloc[0]['EPS'] > 0 else 0
                df_p['Fair'] = df_p['EPS'] * scale
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(df_p.index, df_p['Close'], label='Price')
                ax.plot(df_p.index, df_p['Fair'], label='Fair', linestyle='--')
                apply_strong_style(ax, f"Base: {base_year}", "Price")
                st.pyplot(fig)

elif main_menu == "개별종목 적정주가 분석 2":
    with st.container(border=True):
        v2_ticker = st.text_input("🏢 티커 입력", "PAYX").upper().strip()
        run_v2 = st.button("분석 실행")
    if run_v2 and v2_ticker:
        stock = yf.Ticker(v2_ticker)
        # 분석 2 로직 실행...
        st.info("정밀 가치 분석 데이터를 불러오는 중...")

elif main_menu == "기업 가치 비교 (PER/EPS)":
    with st.container(border=True):
        ticker_input = st.text_input("🏢 티커 입력 (쉼표 구분)", "AAPL, MSFT, NVDA")
        analyze_btn = st.button("비교 분석 실행")
    if analyze_btn:
        st.info("기업 가치 비교 차트를 생성 중입니다...")

else: # ETF 섹터 수익률 분석
    with st.container(border=True):
        selected_etfs = st.multiselect("🌐 ETF 선택", ["SPY", "QQQ", "XLK", "XLE"], default=["SPY", "QQQ"])
        run_etf = st.button("ETF 분석 실행")
    if run_etf:
        st.info("ETF 수익률 데이터를 시각화 중입니다...")
