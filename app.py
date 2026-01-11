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
st.set_page_config(page_title="Professional Stock Analyzer", layout="wide")

# --- [공통 스타일 함수] ---
def apply_strong_style(ax, title, xlabel, ylabel):
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
    
    # X, Y축 라인 생성
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.2)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)

# --- [사이드바 메뉴] ---
with st.sidebar:
    st.title("📂 분석 메뉴")
    main_menu = st.radio(
        "분석 종류를 선택하세요:",
        (
            "개별종목 적정주가 분석 1", 
            "개별종목 적정주가 분석 2", 
            "개별종목 적정주가 분석 3", 
            "기업 가치 비교 (PER/EPS)", 
            "ETF 섹터 수익률 분석"
        )
    )

# --- 메뉴 1 & 2 로직 ---
if main_menu == "개별종목 적정주가 분석 1":
    st.info("개별종목 적정주가 분석 1 화면입니다.")

elif main_menu == "개별종목 적정주가 분석 2":
    st.info("개별종목 적정주가 분석 2 화면입니다.")

# --- [메뉴 3: 개별종목 적정주가 분석 3] ---
elif main_menu == "개별종목 적정주가 분석 3":
    st.title("📈 개별종목 PER 추이 및 평균/중위값 분석")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            v3_ticker = st.text_input("🏢 분석 티커", "MSFT").upper().strip()
        with col2:
            v3_start_year = st.number_input("📅 시작 연도", 2010, 2025, 2017)
        with col3:
            v3_predict_mode = st.radio(
                "🔮 미래 예측 포함 옵션",
                ("None", "현재 분기 예측", "다음 분기 예측"),
                horizontal=True, index=0
            )
        run_v3 = st.button("PER 정밀 분석 실행", type="primary", use_container_width=True)

    if run_v3 and v3_ticker:
        try:
            with st.spinner(f"{v3_ticker} 데이터 분석 중..."):
                # 1. 데이터 수집
                url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                
                target_df = None
                for df in dfs:
                    if df.iloc[:, 0].astype(str).str.contains('PER').any():
                        target_df = df.set_index(df.columns[0])
                        break
                
                per_raw = target_df[target_df.index.str.contains('PER')].transpose()
                eps_raw = target_df[target_df.index.str.contains('EPS')].transpose()
                
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
                plot_df = combined[combined.index >= f"{v3_start_year}-01-01"].copy()

                # 2. 야후 예측치 수집
                stock = yf.Ticker(v3_ticker)
                current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                est = stock.earnings_estimate

                # 3. 미래 예측 로직
                if v3_predict_mode != "None" and est is not None and not est.empty:
                    historical_eps = combined['EPS'].tolist()
                    last_label = plot_df['Label'].iloc[-1]
                    last_yr, last_q = int("20" + last_label.split('.')[0]), int(last_label.split('Q')[1])

                    if v3_predict_mode in ["현재 분기 예측", "다음 분기 예측"]:
                        t1_q, t1_yr = (last_q + 1, last_yr) if last_q < 4 else (1, last_yr + 1)
                        ttm_eps_1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                        plot_df.loc[pd.Timestamp(f"{t1_yr}-{(t1_q-1)*3+1}-01")] = [current_price / ttm_eps_1, np.nan, f"{str(t1_yr)[2:]}.Q{t1_q}(E)"]

                    if v3_predict_mode == "다음 분기 예측":
                        t1_q_tmp, t1_yr_tmp = (last_q + 1, last_yr) if last_q < 4 else (1, last_yr + 1)
                        t2_q, t2_yr = (t1_q_tmp + 1, t1_yr_tmp) if t1_q_tmp < 4 else (1, t1_yr_tmp + 1)
                        ttm_eps_2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                        plot_df.loc[pd.Timestamp(f"{t2_yr}-{(t2_q-1)*3+1}-01")] = [current_price / ttm_eps_2, np.nan, f"{str(t2_yr)[2:]}.Q{t2_q}(E)"]

                avg_per = plot_df['PER'].mean()
                median_per = plot_df['PER'].median()

                # 4. 시각화 (기존 대비 30% 추가 축소: 10.5x4.9 -> 7.5x3.5)
                fig, ax = plt.subplots(figsize=(7.5, 3.5), facecolor='white')
                
                # 라인 그리기
                ax.plot(plot_df['Label'], plot_df['PER'], marker='o', linestyle='-', color='#0047AB', 
                        linewidth=1.8, markersize=5, label='PER Trend')
                ax.axhline(avg_per, color='#D32F2F', linestyle='--', linewidth=1.2, label=f'Average ({avg_per:.2f})')
                ax.axhline(median_per, color='#7B1FA2', linestyle='-.', linewidth=1.2, label=f'Median ({median_per:.2f})')

                # 예측 구간 강조
                for i, label in enumerate(plot_df['Label']):
                    if "(E)" in label:
                        ax.axvspan(i-0.4, i+0.4, color='#FF8C00', alpha=0.1)

                # 스타일 적용 (X, Y축 라벨 추가)
                apply_strong_style(ax, f"[{v3_ticker}] PER Analysis", "Quarter (Time)", "PER Value")
                
                # 범례 설정 (내용 보강 및 배경색 흰색)
                ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d3d3d3', 
                          framealpha=1, fontsize=7, shadow=False)
                
                st.pyplot(fig)

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("현재 주가", f"${current_price:.2f}")
                c2.metric("평균 PER", f"{avg_per:.2f}x")
                c3.metric("중위 PER", f"{median_per:.2f}x")

        except Exception as e:
            st.error(f"오류 발생: {e}")

elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.info("기업 가치 비교 페이지입니다.")
else:
    st.info("ETF 섹터 수익률 분석 페이지입니다.")
