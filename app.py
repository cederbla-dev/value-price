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
def apply_strong_style(ax, title, ylabel):
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)

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

# --- 메뉴 1 & 2는 기존 로직을 유지하거나 생략 (요청하신 분석 3 위주로 기술) ---
if main_menu == "개별종목 적정주가 분석 1":
    st.info("개별종목 적정주가 분석 1 화면입니다. (기존 코드 유지)")

elif main_menu == "개별종목 적정주가 분석 2":
    st.info("개별종목 적정주가 분석 2 화면입니다. (기존 코드 유지)")

# --- [메뉴 3: 개별종목 적정주가 분석 3 (PER Mean vs Median)] ---
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
            with st.spinner(f"{v3_ticker} 데이터 수집 및 TTM 계산 중..."):
                # 1. 과거 데이터 수집
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

                # 분기 라벨 생성 함수
                def get_q_label(dt):
                    year = dt.year if dt.day > 5 else (dt - timedelta(days=5)).year
                    month = dt.month if dt.day > 5 else (dt - timedelta(days=5)).month
                    q = (month-1)//3 + 1
                    return f"{str(year)[2:]}.Q{q}"

                combined['Label'] = [get_q_label(d) for d in combined.index]
                plot_df = combined[combined.index >= f"{v3_start_year}-01-01"].copy()

                # 2. 야후 예측치 및 주가 수집
                stock = yf.Ticker(v3_ticker)
                current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                est = stock.earnings_estimate

                # 3. 미래 예측치(E) 슬라이딩 로직
                if v3_predict_mode != "None" and est is not None and not est.empty:
                    historical_eps = combined['EPS'].tolist()
                    last_label = plot_df['Label'].iloc[-1]
                    last_yr = int("20" + last_label.split('.')[0])
                    last_q = int(last_label.split('Q')[1])

                    # 현재 분기(0q) 예측 추가
                    if v3_predict_mode in ["현재 분기 예측", "다음 분기 예측"]:
                        curr_q_est = est.loc['0q', 'avg']
                        t1_q, t1_yr = (last_q + 1, last_yr) if last_q < 4 else (1, last_yr + 1)
                        label_1 = f"{str(t1_yr)[2:]}.Q{t1_q}(E)"
                        ttm_eps_1 = sum(historical_eps[-3:]) + curr_q_est
                        per_1 = current_price / ttm_eps_1
                        plot_df.loc[pd.Timestamp(f"{t1_yr}-{(t1_q-1)*3+1}-01")] = [per_1, np.nan, label_1]

                    # 다음 분기(+1q) 예측 추가
                    if v3_predict_mode == "다음 분기 예측":
                        next_q_est = est.loc['+1q', 'avg']
                        t2_q, t2_yr = (t1_q + 1, t1_yr) if t1_q < 4 else (1, t1_yr + 1)
                        label_2 = f"{str(t2_yr)[2:]}.Q{t2_q}(E)"
                        ttm_eps_2 = sum(historical_eps[-2:]) + curr_q_est + next_q_est
                        per_2 = current_price / ttm_eps_2
                        plot_df.loc[pd.Timestamp(f"{t2_yr}-{(t2_q-1)*3+1}-01")] = [per_2, np.nan, label_2]

                # 4. 통계치 계산
                per_series = plot_df['PER'].dropna()
                avg_per = per_series.mean()
                median_per = per_series.median()

                # 5. 시각화 (80% 사이즈 최적화)
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                
                # 메인 추이선
                ax.plot(plot_df['Label'], plot_df['PER'], marker='o', linestyle='-', color='#34495e', 
                        linewidth=2, markersize=8, label='Forward PER Trend')

                # 예측 구간 하이라이트
                for i, label in enumerate(plot_df['Label']):
                    if "(E)" in label:
                        ax.axvspan(i-0.4, i+0.4, color='orange', alpha=0.15)
                        ax.text(i, plot_df['PER'].iloc[i] + 0.3, f"{plot_df['PER'].iloc[i]:.2f}", 
                                ha='center', fontweight='bold', color='#d35400')

                # 평균선 및 중위값선
                ax.axhline(avg_per, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Average: {avg_per:.2f}')
                ax.axhline(median_per, color='#8e44ad', linestyle='-.', linewidth=1.5, label=f'Median: {median_per:.2f}')

                apply_strong_style(ax, f"[{v3_ticker}] PER Analysis: Mean vs Median (Since {v3_start_year})", "PER (Price / TTM EPS)")
                ax.legend(loc='upper left', frameon=True, shadow=True)
                
                # 차트 출력
                st.pyplot(fig)

                # 요약 정보 카드
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("현재 주가", f"${current_price:.2f}")
                c2.metric("과거 평균 PER", f"{avg_per:.2f}x")
                c3.metric("과거 중위 PER", f"{median_per:.2f}x")

        except Exception as e:
            st.error(f"데이터를 분석하는 중 오류가 발생했습니다: {e}")

# --- 그 외 메뉴 로직 ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.info("기업 가치 비교 페이지입니다.")

else:
    st.info("ETF 섹터 수익률 분석 페이지입니다.")
