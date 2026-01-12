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

# --- 메뉴 1: 개별종목 적정주가 분석 1 (범례 배경색 및 정렬 최종 수정본) ---
if main_menu == "개별종목 적정주가 분석 1":
    # 1. 상단 입력 UI 레이아웃
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
            # 데이터 가져오기 (기존 정의된 fetch_valuation_data 함수 호출)
            combined = fetch_valuation_data(val_ticker, val_predict_mode)
            
            if combined is not None and not combined.empty:
                final_price = combined['Close'].iloc[-1]
                target_date_label = combined.index[-1]
                summary_list = []

                # --- 파트 A: 연도별 그래프 생성 ---
                st.subheader(f"📈 {val_ticker} 연도별 적정주가 시뮬레이션")
                
                for base_year in range(2017, 2026):
                    df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                    
                    if len(df_plot) < 2 or df_plot.iloc[0]['EPS'] <= 0:
                        continue
                    
                    # 기준 PER 산출 및 적정가(Fair Value) 계산
                    scale_factor = df_plot.iloc[0]['Close'] / df_plot.iloc[0]['EPS']
                    df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                    
                    last_fair_value = df_plot.iloc[-1]['Fair_Value']
                    gap_pct = ((final_price - last_fair_value) / last_fair_value) * 100
                    status = "🔴 고평가" if gap_pct > 0 else "🔵 저평가"

                    # 표 데이터 저장용 리스트업
                    summary_list.append({
                        "기준 연도": f"{base_year}년",
                        "기준 PER": f"{scale_factor:.1f}x",
                        "적정 주가": f"${last_fair_value:.2f}",
                        "현재 주가": f"${final_price:.2f}",
                        "괴리율 (%)": f"{gap_pct:+.1f}%",
                        "상태": status
                    })

                    # 그래프 시각화 설정
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    
                    # 1. Price 라인 (파란색)
                    ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', 
                            linewidth=2.0, marker='o', markersize=4, label='Price')
                    # 2. EPS 가치 라인 (빨간색)
                    ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', 
                            linestyle='--', marker='s', markersize=4, label='EPS')
                    
                    # 미래 예측(Est.) 구간 하이라이트
                    for i, idx in enumerate(df_plot.index):
                        if "(Est.)" in str(idx):
                            ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.1)

                    # 스타일 적용 (기존 apply_strong_style 함수)
                    apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                    plt.xticks(rotation=45)
                    
                    # --- [범례 커스텀 수정] 배경 흰색 및 글자색 지정 ---
                    leg = ax.legend(
                        loc='upper left', 
                        fontsize=11, 
                        frameon=True, 
                        facecolor='white',  # 범례 내부 배경색 흰색
                        edgecolor='black',  # 범례 테두리색 검정
                        framealpha=1.0      # 투명도 없음 (불투명 흰색)
                    )
                    
                    # 범례 내 텍스트 색상 및 굵기 개별 설정
                    for text in leg.get_texts():
                        if text.get_text() == 'Price':
                            text.set_color('#1f77b4')  # 파란색 글씨
                            text.set_weight('bold')
                        elif text.get_text() == 'EPS':
                            text.set_color('#d62728')  # 빨간색 글씨
                            text.set_weight('bold')
                    
                    st.pyplot(fig)
                    plt.close(fig)

                # --- 파트 B: 최종 요약 표 출력 (60% 너비 및 왼쪽 정렬) ---
                if summary_list:
                    st.write("\n")
                    st.markdown("---")
                    st.subheader(f"📊 {val_ticker} 밸류에이션 종합 요약")
                    st.caption(f"분석 기준점(Target Date): {target_date_label}")

                    summary_df = pd.DataFrame(summary_list)

                    # 표의 시작점을 그래프의 왼쪽 끝과 맞추기 위해 6:4 비율 컬럼 사용
                    main_col, _ = st.columns([6, 4]) 
                    
                    with main_col:
                        st.dataframe(
                            summary_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "기준 연도": st.column_config.TextColumn("기준 연도"),
                                "기준 PER": st.column_config.TextColumn("기준 PER"),
                                "적정 주가": st.column_config.TextColumn("적정 주가"),
                                "현재 주가": st.column_config.TextColumn("현재 주가"),
                                "괴리율 (%)": st.column_config.TextColumn("괴리율 (%)"),
                                "상태": st.column_config.TextColumn("상태"),
                            }
                        )
                    
                    st.info(f"💡 **분석 가이드**: 다수의 기준 연도 대비 '저평가'가 많다면 현재 주가는 매력적인 구간일 확률이 높습니다.")
                else:
                    st.warning("분석 가능한 흑자(EPS > 0) 데이터가 부족합니다.")
            else:
                st.error("데이터를 수집하지 못했습니다. 티커 입력이 정확한지 확인해 주세요.")

# --- 메뉴 2: 개별종목 적정주가 분석 2 (UI 비율 및 버튼명 수정 버전) ---
if main_menu == "개별종목 적정주가 분석 2":
    # 1. 상단 입력 UI 레이아웃 (전체 너비의 50%씩 배분하여 왼쪽 정렬)
    with st.container(border=True):
        # 입력창과 버튼을 묶어서 왼쪽으로 배치하기 위해 5:5 비율의 컬럼 생성
        col1, col2 = st.columns([1, 1])  
        
        with col1:
            val_ticker_2 = st.text_input("🏢 분석 티커 입력", "TSLA", key="ticker_2").upper().strip()
        
        with col2:
            # 입력창의 라벨 높이만큼 공간을 띄워 버튼과 입력창의 높이를 맞춤
            st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            run_val_2 = st.button("당해 EPS 기반 가치 분석", type="primary", use_container_width=True)

    if run_val_2 and val_ticker_2:
        with st.spinner(f"[{val_ticker_2}] 당해 실적 기반 정밀 가치 분석을 진행 중입니다..."):
            # 데이터 수집 (기존 정의된 fetch_valuation_data 활용, 예측 옵션은 기본값)
            combined = fetch_valuation_data(val_ticker_2, "None")
            
            if combined is not None and not combined.empty:
                # 당해(가장 최근 확정 실적 시점) 데이터 추출
                final_price = combined['Close'].iloc[-1]
                target_date_label = combined.index[-1]
                
                # 분석을 위한 리스트 (메뉴 1과 동일한 로직의 요약 데이터 생성)
                summary_list_2 = []

                # --- 파트 A: 시각화 로직 ---
                st.subheader(f"📊 {val_ticker_2} 당해 EPS 기반 적정주가 분석")
                
                # 역사적 기준점들을 순회하며 당해 가격과 비교
                for base_year in range(2017, 2026):
                    df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                    
                    if len(df_plot) < 2 or df_plot.iloc[0]['EPS'] <= 0:
                        continue
                    
                    # 기준 PER 산출 및 적정가 계산
                    scale_factor = df_plot.iloc[0]['Close'] / df_plot.iloc[0]['EPS']
                    df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                    
                    last_fair_value = df_plot.iloc[-1]['Fair_Value']
                    gap_pct = ((final_price - last_fair_value) / last_fair_value) * 100
                    status = "🔴 고평가" if gap_pct > 0 else "🔵 저평가"

                    summary_list_2.append({
                        "기준 연도": f"{base_year}년",
                        "기준 PER": f"{scale_factor:.1f}x",
                        "적정 주가": f"${last_fair_value:.2f}",
                        "현재 주가": f"${final_price:.2f}",
                        "괴리율 (%)": f"{gap_pct:+.1f}%",
                        "상태": status
                    })

                    # 그래프 시각화 (범례 및 스타일 적용)
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    
                    ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', 
                            linewidth=2.0, marker='o', markersize=4, label='Price')
                    ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', 
                            linestyle='--', marker='s', markersize=4, label='EPS')

                    apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                    plt.xticks(rotation=45)
                    
                    # 범례 설정 (흰색 배경 및 커스텀 색상)
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

                # --- 파트 B: 최종 요약 표 출력 (60% 너비, 왼쪽 정렬) ---
                if summary_list_2:
                    st.write("\n")
                    st.markdown("---")
                    st.subheader(f"📋 {val_ticker_2} 가치 분석 요약")
                    
                    summary_df_2 = pd.DataFrame(summary_list_2)
                    
                    # 왼쪽 정렬을 위한 6:4 비율 컬럼
                    main_col_2, _ = st.columns([6, 4]) 
                    
                    with main_col_2:
                        st.dataframe(
                            summary_df_2,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "기준 연도": st.column_config.TextColumn("기준 연도"),
                                "기준 PER": st.column_config.TextColumn("기준 PER"),
                                "적정 주가": st.column_config.TextColumn("적정 주가"),
                                "현재 주가": st.column_config.TextColumn("현재 주가"),
                                "괴리율 (%)": st.column_config.TextColumn("괴리율 (%)"),
                                "상태": st.column_config.TextColumn("상태"),
                            }
                        )
                    st.info("💡 본 분석은 당해 확정 실적(EPS)만을 기준으로 산출된 보수적 가치 평가 결과입니다.")
                else:
                    st.warning("분석 가능한 충분한 데이터가 확보되지 않았습니다.")
            else:
                st.error("데이터 수집에 실패했습니다. 티커명을 확인해 주세요.")

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
