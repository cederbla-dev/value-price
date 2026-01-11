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

# --- [공통] 스타일 및 유틸리티 함수 ---
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

def normalize_to_standard_quarter(dt):
    """서로 다른 분기 마감일을 가장 가까운 표준 분기(3, 6, 9, 12월)로 조정"""
    month = dt.month
    year = dt.year
    if month in [1, 2, 3]:   new_month, new_year = 3, year
    elif month in [4, 5, 6]: new_month, new_year = 6, year
    elif month in [7, 8, 9]: new_month, new_year = 9, year
    elif month in [10, 11, 12]: new_month, new_year = 12, year
    return pd.Timestamp(year=new_year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

# --- [데이터 처리 함수들] ---

@st.cache_data(ttl=3600)
def fetch_multicycle_ticker_per(ticker, predict_mode):
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

        # 미래 예측치 계산
        if predict_mode != "None":
            stock = yf.Ticker(ticker)
            current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
            est = stock.earnings_estimate
            
            if est is not None and not est.empty:
                last_dt = combined.index[-1]
                historical_eps = combined['EPS'].tolist()
                
                # Q1 예측
                q1_dt = last_dt + pd.DateOffset(months=3)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1

                # Q2 예측
                if predict_mode == "다음 분기 예측":
                    q2_dt = q1_dt + pd.DateOffset(months=3)
                    ttm_eps_q2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[q2_dt, 'PER'] = current_price / ttm_eps_q2

        # 표준 분기로 날짜 스냅
        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined['PER']
    except:
        return None

# (기존 fetch_valuation_data_logic_1 함수 등은 생략/유지)

# --- [UI 레이아웃] ---

with st.sidebar:
    st.title("📂 분석 메뉴")
    main_menu = st.radio(
        "분석 종류를 선택하세요:",
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

# --- 메뉴 1 & 2 로직 (이전 대화에서 제공된 코드 유지) ---
if main_menu == "개별종목 적정주가 분석 1":
    st.title(f"🚀 {main_menu}")
    # ... (기존 분석 1 코드) ...
    st.info("개별종목 적정주가 분석 1 화면입니다.")

elif main_menu == "개별종목 적정주가 분석 2":
    st.title(f"🚀 {main_menu}")
    # ... (기존 분석 2 코드) ...
    st.info("개별종목 적정주가 분석 2 화면입니다.")

# --- 메뉴 3: 개별종목 적정주가 분석 3 (다중 종목 회계 주기 동기화 분석) ---
elif main_menu == "개별종목 적정주가 분석 3":
    st.title("🔄 회계 주기 동기화 PER 추이 비교")
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            v3_tickers = st.text_input("🏢 비교 종목 입력 (예: AAPL, AVGO, NKE)", "AAPL, AVGO, NKE").upper().replace(',', ' ').split()
        with col2:
            v3_start_year = st.number_input("📅 기준 연도", 2010, 2025, 2017)
        with col3:
            # 미래 예측 옵션 (Default: None)
            v3_predict_mode = st.radio(
                "🔮 미래 예측 옵션",
                ("None", "현재 분기 예측", "다음 분기 예측"),
                horizontal=True, index=0
            )
        run_v3 = st.button("동기화 분석 실행", type="primary", use_container_width=True)

    if run_v3 and v3_tickers:
        with st.spinner("회계 주기 동기화 및 데이터 분석 중..."):
            master_df = pd.DataFrame()
            for ticker in v3_tickers:
                series = fetch_multicycle_ticker_per(ticker, v3_predict_mode)
                if series is not None:
                    master_df[ticker] = series
            
            if not master_df.empty:
                master_df = master_df[master_df.index >= f"{v3_start_year}-01-01"].sort_index()
                # 첫 번째 유효 행을 기준으로 Index 100화
                indexed_df = (master_df / master_df.apply(lambda x: x.dropna().iloc[0])) * 100
                
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                
                x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_df.index]
                x_indices = np.arange(len(indexed_df))
                
                forecast_count = 1 if v3_predict_mode == "현재 분기 예측" else (2 if v3_predict_mode == "다음 분기 예측" else 0)
                
                for ticker in indexed_df.columns:
                    series = indexed_df[ticker].dropna()
                    valid_indices = [indexed_df.index.get_loc(dt) for dt in series.index]
                    
                    # 과거 데이터와 예측 데이터 분리
                    if forecast_count > 0:
                        hist_idx = valid_indices[:-forecast_count]
                        hist_val = series.values[:-forecast_count]
                        pred_idx = valid_indices[-forecast_count-1:]
                        pred_val = series.values[-forecast_count-1:]
                        
                        line, = ax.plot(hist_idx, hist_val, marker='o', label=f"{ticker} (Idx: {series.iloc[-1]:.1f})", linewidth=2)
                        ax.plot(pred_idx, pred_val, linestyle='--', color=line.get_color(), alpha=0.7)
                        ax.scatter(valid_indices[-forecast_count:], series.values[-forecast_count:], marker='D', s=50, color=line.get_color(), zorder=5)
                    else:
                        ax.plot(valid_indices, series.values, marker='o', label=f"{ticker} (Idx: {series.iloc[-1]:.1f})", linewidth=2)

                apply_strong_style(ax, f"Multi-Cycle PER Trend (Base 100 at {v3_start_year})", "Relative PER Index")
                ax.axhline(100, color='black', linewidth=1, alpha=0.5)
                ax.set_xticks(x_indices)
                ax.set_xticklabels(x_labels, rotation=45)
                ax.legend(loc='upper left', frameon=True)
                
                st.pyplot(fig)
                
                st.info("💡 **분석 가이드**: 실선은 확정 실적 기반 PER이며, 점선과 다이아몬드 마커는 야후 컨센서스 예측치(TTM)가 반영된 PER입니다. 모든 데이터는 달력상 표준 분기(3, 6, 9, 12월)로 동기화되었습니다.")
            else:
                st.error("데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")

# --- (이하 기존 메뉴 4, 5 로직 유지) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.info("기업 가치 비교 페이지입니다.")

else:
    st.info("ETF 섹터 수익률 분석 페이지입니다.")
