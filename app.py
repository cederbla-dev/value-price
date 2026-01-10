import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import io
import numpy as np
from datetime import datetime
import warnings

# 1. 환경 설정 및 경고 무시
warnings.filterwarnings("ignore")
st.set_page_config(page_title="미국주식 통합 분석 대시보드", layout="wide")

# 한글 깨짐 방지 설정 (Streamlit 환경은 보통 영문 폰트이므로 기본 폰트 사용)
mpl.rcParams['axes.unicode_minus'] = False

# ==========================================
# [공통 함수] 데이터 수집 및 유틸리티 (캐싱 및 방어로직 강화)
# ==========================================

@st.cache_data(ttl=3600)
def get_choicestock_data(ticker, data_type='EPS'):
    """
    ChoiceStock에서 과거 실적 데이터를 크롤링합니다.
    방어 로직: 헤더 추가, 여러 파서 시도, 예외 처리 강화
    """
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    # 실제 브라우저처럼 보이기 위한 User-Agent 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # pandas의 read_html이 lxml이나 html5lib를 사용하도록 설정
        # io.StringIO를 사용하여 직접 텍스트를 전달
        dfs = pd.read_html(io.StringIO(response.text))
        
        target_df = None
        for df in dfs:
            if df.shape[1] > 0 and df.iloc[:, 0].astype(str).str.contains(data_type).any():
                target_df = df.set_index(df.columns[0])
                break
        
        if target_df is None:
            return pd.DataFrame()

        # 데이터 전처리: 행/열 전환 및 날짜 인덱스화
        raw_data = target_df[target_df.index.str.contains(data_type, na=False)].transpose()
        raw_data.index = pd.to_datetime(raw_data.index, format='%y.%m.%d', errors='coerce')
        raw_data = raw_data.dropna().sort_index()
        
        # 숫자 변환 (콤마 제거)
        col_name = 'Value'
        raw_data.columns = [col_name]
        raw_data[col_name] = pd.to_numeric(raw_data[col_name].astype(str).str.replace(',', ''), errors='coerce')
        
        # 2017년 1월 1일 이후 데이터만 유지 (사용자 요청 사항)
        raw_data = raw_data[raw_data.index >= "2017-01-01"]
        
        return raw_data

    except Exception as e:
        st.error(f"{ticker}의 {data_type} 데이터를 가져오는 중 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yahoo_price(ticker):
    """2017년부터 현재까지의 주가 수집"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start="2017-01-01")
        if df.empty:
            return pd.Series()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df['Close']
    except:
        return pd.Series()

@st.cache_data(ttl=3600)
def get_yahoo_estimates(ticker):
    """Yahoo Finance 향후 2분기 EPS 예측치 수집"""
    try:
        stock = yf.Ticker(ticker)
        est = stock.earnings_estimate
        if est is not None and not est.empty:
            # avg 열의 0q(현재분기), +1q(다음분기), 0y(올해연간)
            return {
                'curr_q': est.loc['0q', 'avg'] if '0q' in est.index else None,
                'next_q': est.loc['+1q', 'avg'] if '+1q' in est.index else None,
                'curr_y': est.loc['0y', 'avg'] if '0y' in est.index else None
            }
    except:
        pass
    return {}

# ==========================================
# [화면 구성] 모듈별 렌더링 함수
# ==========================================

def render_valuation_master():
    st.subheader("💎 개별 종목 정밀 밸류에이션")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker = st.text_input("티커(Ticker) 입력", value="AAPL").upper().strip()
        include_est = st.selectbox("미래 예측치 포함", ["포함 안 함", "현재 분기만", "다음 분기까지"])
    
    if not ticker: return

    # 데이터 로드
    with st.spinner(f"{ticker} 데이터를 분석 중입니다..."):
        eps_data = get_choicestock_data(ticker, 'EPS')
        price_data = get_yahoo_price(ticker)
        estimates = get_yahoo_estimates(ticker)

    if eps_data.empty or price_data.empty:
        st.warning("데이터를 불러올 수 없습니다. 티커가 정확한지 확인해 주세요.")
        return

    # 분석 로직 통합
    tab1, tab2, tab3 = st.tabs(["연도별 적정주가", "PER 밴드", "PEG 분석"])

    with tab1:
        # 연도별 기준점 밸류에이션 (File 6 로직)
        st.write("### 연도별 시작점 기준 적정주가 판단")
        
        # 월말 주가로 리샘플링하여 EPS와 날짜 맞춤
        price_m = price_data.resample('M').last()
        combined = pd.DataFrame({'EPS': eps_data['Value']})
        combined = combined.join(price_m, how='inner').dropna()
        
        # 예측치 추가
        if include_est != "포함 안 함":
            last_date = combined.index[-1]
            curr_p = combined['Close'].iloc[-1]
            if estimates.get('curr_q'):
                combined.loc[last_date + pd.DateOffset(months=3)] = [estimates['curr_q'], curr_p]
            if include_est == "다음 분기까지" and estimates.get('next_q'):
                combined.loc[last_date + pd.DateOffset(months=6)] = [estimates['next_q'], curr_p]

        results = []
        for year in range(2017, datetime.now().year + 1):
            start_key = f"{year}-01"
            subset = combined[combined.index >= start_key].copy()
            if len(subset) < 2: continue
            
            base_eps = subset.iloc[0]['EPS']
            base_price = subset.iloc[0]['Close']
            if base_eps <= 0: continue
            
            per_factor = base_price / base_eps
            subset['Fair_Value'] = subset['EPS'] * per_factor
            
            final_actual = subset['Close'].iloc[-1]
            final_fair = subset['Fair_Value'].iloc[-1]
            gap = ((final_actual - final_fair) / final_fair) * 100
            
            results.append({
                "기준 연도": year,
                "기준 PER": round(per_factor, 1),
                "현재 적정가": round(final_fair, 2),
                "괴리율(%)": round(gap, 2),
                "상태": "고평가" if gap > 0 else "저평가"
            })

        st.table(pd.DataFrame(results))

    with tab2:
        # PER 추세 분석 (File 8 로직)
        per_df = get_choicestock_data(ticker, 'PER')
        if not per_df.empty:
            avg_per = per_df['Value'].mean()
            med_per = per_df['Value'].median()
            curr_per = per_df['Value'].iloc[-1]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(per_df.index, per_df['Value'], marker='o', label='PER Trend')
            ax.axhline(avg_per, color='red', linestyle='--', label=f'Mean: {avg_per:.2f}')
            ax.axhline(med_per, color='purple', linestyle='-.', label=f'Median: {med_per:.2f}')
            ax.set_title(f"{ticker} PER Band (Current: {curr_per:.2f})")
            ax.legend()
            st.pyplot(fig)

    with tab3:
        # PEG 분석 (File 11 로직)
        if len(eps_data) >= 8:
            ttm_now = eps_data['Value'].iloc[-4:].sum()
            curr_price = price_data.iloc[-1]
            curr_per = curr_price / ttm_now
            
            # 3년전 TTM 대비 성장률
            ttm_3y_ago = eps_data['Value'].iloc[-12:-8].sum()
            if ttm_3y_ago > 0:
                cagr = ((ttm_now / ttm_3y_ago) ** (1/3) - 1) * 100
                peg = curr_per / cagr if cagr > 0 else 0
                st.metric("3년 CAGR 기준 PEG", round(peg, 2), f"{cagr:.1f}% 성장")

def render_market_analysis():
    st.subheader("📊 섹터 및 지수 성과 비교")
    all_tickers = ["SPY", "QQQ", "XLK", "XLV", "XLY", "XLF", "XLI", "XLP", "XLE", "XLC", "XLB", "XLU", "XLRE"]
    selected = st.multiselect("비교 대상 선택", all_tickers, default=["SPY", "QQQ", "XLK"])
    
    start_date = st.date_input("비교 시작일", value=datetime(2017, 1, 1))
    
    if st.button("수익률 차트 생성"):
        combined_price = pd.DataFrame()
        for t in selected:
            s = yf.Ticker(t).history(start=start_date)['Close']
            if not s.empty:
                # 시작 시점 100으로 정규화
                combined_price[t] = (s / s.iloc[0]) * 100
        
        if not combined_price.empty:
            st.line_chart(combined_price)

# ==========================================
# 메인 실행부
# ==========================================
def main():
    st.sidebar.title("Stock Dashboard")
    page = st.sidebar.selectbox("메뉴 선택", ["홈", "종목 밸류에이션", "시장 성과 비교"])
    
    if page == "홈":
        st.title("🍎 미국주식 투자 분석 통합 앱")
        st.write("왼쪽 사이드바에서 메뉴를 선택해 주세요.")
        st.info("모든 분석은 2017년 1월 1일 이후 데이터를 기준으로 합니다.")
    elif page == "종목 밸류에이션":
        render_valuation_master()
    elif page == "시장 성과 비교":
        render_market_analysis()

if __name__ == "__main__":
    main()
