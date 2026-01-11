import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import warnings

# 기본 설정 및 경고 무시
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Global Stock PER Analyzer", layout="wide")

def normalize_to_standard_quarter(dt):
    """서로 다른 분기 마감일을 가장 가까운 표준 분기(3, 6, 9, 12월)로 조정"""
    month = dt.month
    year = dt.year
    if month in [1, 2, 3]:   new_month, new_year = 3, year
    elif month in [4, 5, 6]: new_month, new_year = 6, year
    elif month in [7, 8, 9]: new_month, new_year = 9, year
    elif month in [10, 11, 12]: new_month, new_year = 12, year
    return pd.Timestamp(year=new_year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

@st.cache_data(ttl=3600)  # 1시간 동안 데이터 캐싱
def fetch_multicycle_ticker_per(ticker, show_q1, show_q2):
    """다양한 회계 주기를 처리하는 검증된 PER 추출 함수"""
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
            # 최신 가격 가져오기 로직 보강
            history = stock.history(period="1d")
            current_price = history['Close'].iloc[-1] if not history.empty else 0
            
            est = stock.earnings_estimate
            
            if est is not None and not est.empty:
                last_dt = combined.index[-1]
                
                # Q1 예측
                q1_dt = last_dt + pd.DateOffset(months=3)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1

                # Q2 예측
                if show_q2:
                    q2_dt = q1_dt + pd.DateOffset(months=3)
                    ttm_eps_q2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[q2_dt, 'PER'] = current_price / ttm_eps_q2

        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()

        return combined['PER']
    except Exception as e:
        return None

# --- UI 레이아웃 ---
st.title("📊 회계 주기 동기화 PER 트렌드 분석기")
st.markdown("""
기업마다 다른 **회계 결산일(Fiscal Year End)**을 표준 분기(3, 6, 9, 12월)로 자동 보정하여 
동일 선상에서 밸류에이션 추이를 비교합니다.
""")

with st.sidebar:
    st.header("설정 패널")
    ticker_input = st.text_input("비교 종목 입력 (쉼표 또는 공백 구분)", "AAPL, AVGO, TSLA, NKE")
    start_year = st.number_input("기준 연도", min_value=2010, max_value=2025, value=2017)
    
    st.subheader("예측 데이터(Forward)")
    ans1 = st.checkbox("현재 분기(Q1) 예측 포함", value=True)
    ans2 = st.checkbox("다음 분기(Q2) 예측 포함", value=False)
    
    analyze_btn = st.button("데이터 분석 시작", type="primary")

if analyze_btn:
    tickers = ticker_input.upper().replace(',', ' ').split()
    master_df = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(tickers):
        status_text.text(f"분석 중: {ticker}...")
        series = fetch_multicycle_ticker_per(ticker, ans1, ans2)
        if series is not None:
            master_df[ticker] = series
        progress_bar.progress((idx + 1) / len(tickers))

    if not master_df.empty:
        status_text.text("그래프 생성 중...")
        
        # 데이터 필터링 및 인덱스화
        master_df = master_df[master_df.index >= f"{start_year}-01-01"].sort_index()
        if master_df.empty or master_df.iloc[0].isnull().any():
            st.error("데이터가 부족하거나 기준 시점에 값이 없습니다. 기준 연도를 조정해보세요.")
        else:
            indexed_df = (master_df / master_df.iloc[0]) * 100
            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 7))
            plt.style.use('dark_background') # 웹 다크모드 대응 스타일
            fig.patch.set_facecolor('#0E1117') # Streamlit 배경색과 매칭
            ax.set_facecolor('#0E1117')
            
            x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_df.index]
            x_indices = np.arange(len(indexed_df))

            for ticker in indexed_df.columns:
                series = indexed_df[ticker].dropna()
                forecast_count = (1 if ans1 else 0) + (1 if ans2 else 0)
                
                # 예측치를 포함한 유효 인덱스 매핑
                valid_indices = [indexed_df.index.get_loc(dt) for dt in series.index]
                
                if len(valid_indices) > forecast_count:
                    hist_idx = valid_indices[:-forecast_count] if forecast_count > 0 else valid_indices
                    hist_val = series.values[:-forecast_count] if forecast_count > 0 else series.values
                    
                    line, = ax.plot(hist_idx, hist_val, marker='o', label=f"{ticker} (최종: {series.iloc[-1]:.1f})", linewidth=2)
                    
                    if forecast_count > 0:
                        pred_idx = valid_indices[-forecast_count-1:]
                        pred_val = series.values[-forecast_count-1:]
                        ax.plot(pred_idx, pred_val, linestyle='--', color=line.get_color(), alpha=0.6)
                        ax.scatter(valid_indices[-forecast_count:], series.values[-forecast_count:], 
                                   marker='D', s=50, color=line.get_color(), zorder=5)

            ax.axhline(100, color='white', alpha=0.3, linestyle=':')
            ax.set_title(f"Relative PER Trend (Base 100 at {start_year})", fontsize=15, color='white')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(x_labels, rotation=45, fontsize=9, color='white')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#1E1E1E', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.1)
            
            st.pyplot(fig)
            
            # 데이터 표 출력
            with st.expander("상세 데이터 보기"):
                st.dataframe(master_df.style.highlight_max(axis=0))
                
        status_text.text("분석 완료!")
    else:
        st.error("데이터를 불러오지 못했습니다. 종목 코드를 확인해주세요.")
