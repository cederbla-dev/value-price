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

# --- [UI 레이아웃] ---

with st.sidebar:
    st.title("📂 분석 메뉴")
    main_menu = st.radio(
        "분석 종류를 선택하세요:",
        (
            "개별종목 적정주가 분석 1", 
            "개별종목 적정주가 분석 2", 
            "개별종목 적정주가 분석 3", 
            "개별종목 적정주가 분석 4", 
            "개별종목 적정주가 분석 5", # 신규 추가
            "기업 가치 비교 (PER/EPS)", 
            "ETF 섹터 수익률 분석"
        )
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 1 (기존 코드 유지) ---
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
                    if len(df_plot) < 2 or df_plot.iloc[0]['EPS'] <= 0:
                        continue
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
                    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
                    ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', linewidth=2.0, marker='o', label='Price')
                    ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', marker='s', label='EPS Value')
                    apply_strong_style(ax, f"Base Year: {base_year}", "Price ($)")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close(fig)

                if summary_list:
                    st.markdown("---")
                    st.subheader("📊 종합 요약")
                    st.dataframe(pd.DataFrame(summary_list), use_container_width=True, hide_index=True)

# --- 신규 메뉴: 개별종목 적정주가 분석 5 (복합 가치 분석 모델) ---
elif main_menu == "개별종목 적정주가 분석 5":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        v5_ticker = col1.text_input("🏢 분석 티커 입력", "NVDA").upper().strip()
        v5_discount_rate = col2.slider("📉 기대 수익률(할인율 %)", 5.0, 15.0, 10.0, 0.5)
        run_v5 = st.button("복합 가치 모델 분석 실행", type="primary", use_container_width=True)

    if run_v5 and v5_ticker:
        try:
            with st.spinner('재무 제표 및 가치 분석 데이터를 수집 중입니다...'):
                stock = yf.Ticker(v5_ticker)
                info = stock.info
                
                # 1. 데이터 수집 (자산 가치 및 수익 가치)
                current_price = info.get('currentPrice', 0)
                book_value = info.get('bookValue', 0)  # 주당 순자산
                roe = info.get('returnOnEquity', 0)    # ROE
                eps_forward = info.get('forwardEps', 0) # 예상 EPS
                
                if book_value == 0 or current_price == 0:
                    st.error("분석에 필요한 재무 데이터(BPS, 주가 등)가 부족합니다.")
                else:
                    # 2. 분석 모델링
                    # 모델 A: S-RIM (Residual Income Model) 방식
                    # 적정주가 = BPS + (BPS * (ROE - 할인율) / 할인율)
                    k = v5_discount_rate / 100
                    srim_fair_value = book_value * (roe / k) if roe > 0 else book_value
                    
                    # 모델 B: 수익가치 모델 (EPS x 타겟 PER)
                    target_per = 1 / k  # 할인율의 역수를 적정 PER로 가정
                    earnings_fair_value = eps_forward * target_per
                    
                    # 모델 C: 복합 가치 (자산 40% + 수익 60%)
                    combined_fair_value = (book_value * 0.4) + (earnings_fair_value * 0.6)

                    # 3. 결과 출력
                    st.subheader(f"🔍 [{v5_ticker}] 복합 가치 분석 리포트")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("현재 주가", f"${current_price:.2f}")
                    m2.metric("주당 순자산(BPS)", f"${book_value:.2f}")
                    m3.metric("자기자본이익률(ROE)", f"{roe*100:.2f}%")

                    # 4. 가치 비교 테이블
                    valuation_data = [
                        {"항목": "S-RIM (자산+초과이익)", "계산된 가치": f"${srim_fair_value:.2f}", "현재가 대비": f"{((current_price/srim_fair_value)-1)*100:+.1f}%"},
                        {"항목": "수익 가치 (Forward EPS)", "계산된 가치": f"${earnings_fair_value:.2f}", "현재가 대비": f"{((current_price/earnings_fair_value)-1)*100:+.1f}%"},
                        {"항목": "복합 가치 (종합 판단)", "계산된 가치": f"${combined_fair_value:.2f}", "현재가 대비": f"{((current_price/combined_fair_value)-1)*100:+.1f}%"}
                    ]
                    
                    st.write("### 📊 모델별 적정 주가 비교")
                    st.table(pd.DataFrame(valuation_data))

                    # 5. 시각화 (도넛 차트 - 현재 주가 위치)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    labels = ['S-RIM Value', 'Earnings Value', 'Combined Value', 'Current Price']
                    values = [srim_fair_value, earnings_fair_value, combined_fair_value, current_price]
                    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
                    
                    bars = ax.barh(labels, values, color=colors)
                    ax.axvline(current_price, color='red', linestyle='--', alpha=0.5)
                    apply_strong_style(ax, "Value Comparison ($)", "Model Type")
                    st.pyplot(fig)

                    st.info(f"💡 **분석 결과**: 종합 가치(`${combined_fair_value:.2f}`) 대비 현재 주가는 "
                            f"{'고평가' if current_price > combined_fair_value else '저평가'} 상태입니다.")

        except Exception as e:
            st.error(f"데이터 분석 중 오류 발생: {e}")

# --- 나머지 메뉴 (기존 로직 유지 - 공간상 요약) ---
elif main_menu == "개별종목 적정주가 분석 2":
    st.write("메뉴 2 분석 기능 실행 중...")
    # ... (기존 메뉴 2 코드 삽입)
elif main_menu == "개별종목 적정주가 분석 3":
    st.write("메뉴 3 분석 기능 실행 중...")
    # ... (기존 메뉴 3 코드 삽입)
elif main_menu == "개별종목 적정주가 분석 4":
    st.write("메뉴 4 분석 기능 실행 중...")
    # ... (기존 메뉴 4 코드 삽입)
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.write("기업 가치 비교 분석 실행 중...")
elif main_menu == "ETF 섹터 수익률 분석":
    st.write("ETF 분석 실행 중...")
