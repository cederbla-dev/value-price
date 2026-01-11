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
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15, color='#333333')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=9)

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

# --- 메뉴 1: 개별종목 적정주가 분석 1 ---
if main_menu == "개별종목 적정주가 분석 1":
    st.title("🚀 개별종목 적정주가 분석 1")
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            val_ticker = st.text_input("🏢 분석 티커", "TSLA").upper().strip()
        with col2:
            val_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_val = st.button("적정주가 분석 실행", type="primary", use_container_width=True)

    if run_val and val_ticker:
        try:
            url = f"https://www.choicestock.co.kr/search/invest/{val_ticker}/MRQ"
            dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
            eps_df = pd.DataFrame()
            for df in dfs:
                if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                    target = df.set_index(df.columns[0]).transpose()
                    eps_df = target.iloc[:, [0]].copy()
                    eps_df.columns = ['EPS']
                    break
            eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
            eps_df = eps_df.dropna()
            eps_df.index = [(d.replace(day=1) - timedelta(days=1)).strftime('%Y-%m') if d.day <= 5 else d.strftime('%Y-%m') for d in eps_df.index]
            eps_df['EPS'] = pd.to_numeric(eps_df['EPS'].astype(str).str.replace(',', ''), errors='coerce')
            
            stock = yf.Ticker(val_ticker)
            price_df = stock.history(start="2017-01-01", interval="1mo", auto_adjust=False)
            price_df.index = price_df.index.tz_localize(None).strftime('%Y-%m')
            price_df = price_df[['Close']].copy()
            price_df = price_df[~price_df.index.duplicated(keep='last')]
            combined = pd.merge(eps_df, price_df, left_index=True, right_index=True, how='inner').sort_index()

            summary_data = []
            st.subheader(f"📊 {val_ticker} 연도별 시뮬레이션")
            for base_year in range(2017, 2026):
                df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                if len(df_plot) < 2: continue
                scale_factor = df_plot.iloc[0]['Close'] / df_plot.iloc[0]['EPS']
                df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                gap_pct = ((df_plot.iloc[-1]['Close'] - df_plot.iloc[-1]['Fair_Value']) / df_plot.iloc[-1]['Fair_Value']) * 100
                
                summary_data.append({"Base Year": base_year, "Multiplier": f"{scale_factor:.1f}x", "Gap (%)": f"{gap_pct:+.2f}%"})

                fig, ax = plt.subplots(figsize=(7.7, 3.2))
                ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', linewidth=2, label='Price')
                ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', label='Fair Value')
                apply_strong_style(ax, f"Base Year: {base_year}", "Price ($)")
                st.pyplot(fig)
        except: st.error("데이터 수집 오류")

# --- 메뉴 2: 개별종목 적정주가 분석 2 ---
elif main_menu == "개별종목 적정주가 분석 2":
    st.title("📅 발표일 기준 4분기 단위 분석")
    with st.container(border=True):
        v2_ticker = st.text_input("🏢 분석 티커 입력", "AAPL").upper().strip()
        run_v2 = st.button("과거 기록 기반 분석 시작", type="primary", use_container_width=True)

    if run_v2 and v2_ticker:
        try:
            stock = yf.Ticker(v2_ticker)
            url = f"https://www.choicestock.co.kr/search/invest/{v2_ticker}/MRQ"
            dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
            raw_eps = pd.DataFrame()
            for df in dfs:
                if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                    target_df = df.set_index(df.columns[0])
                    raw_eps = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                    raw_eps.index = pd.to_datetime(raw_eps.index, format='%y.%m.%d').tz_localize(None)
                    raw_eps.columns = ['EPS']
                    break
            raw_eps = raw_eps[raw_eps.index >= "2017-01-01"].sort_index()
            price_df = stock.history(start="2017-01-01", interval="1d")['Close'].tz_localize(None)
            
            current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
            est = stock.earnings_estimate
            current_q_est = est['avg'].iloc[0] if est is not None else 0
            final_target_eps = raw_eps['EPS'].iloc[-3:].sum() + current_q_est

            table_list = []
            for i in range(0, len(raw_eps) - 3, 4):
                group = raw_eps.iloc[i:i+4]
                eps_sum, avg_price = group['EPS'].sum(), price_df[group.index[0]:group.index[-1]].mean()
                if (i + 4 >= len(raw_eps)): eps_sum = final_target_eps
                per = avg_price / eps_sum
                fair_price = final_target_eps * per
                diff_pct = ((current_price / fair_price) - 1) * 100
                table_list.append({
                    '기준 연도': f"{group.index[0].year}년", '4분기 EPS합': f"{eps_sum:.2f}",
                    '과거 PER': f"{per:.1f}x", '적정 가치': f"${fair_price:.2f}", 
                    '판단': f"{abs(diff_pct):.1f}% " + ("저평가" if current_price < fair_price else "고평가")
                })
            st.subheader("📋 과거 4분기 단위 밸류에이션 기록")
            st.dataframe(pd.DataFrame(table_list), width=650, hide_index=True)
        except: st.error("분석 중 오류 발생")

# --- 메뉴 3: 개별종목 적정주가 분석 3 (수정 반영) ---
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
            with st.spinner("데이터 분석 중..."):
                url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
                dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
                target_df = next(df.set_index(df.columns[0]) for df in dfs if 'PER' in df.iloc[:, 0].values)
                
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
                    return f"{str(year)[2:]}.Q{(month-1)//3 + 1}"

                combined['Label'] = [get_q_label(d) for d in combined.index]
                plot_df = combined[combined.index >= f"{v3_start_year}-01-01"].copy()

                stock = yf.Ticker(v3_ticker)
                current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                est = stock.earnings_estimate

                if v3_predict_mode != "None" and est is not None:
                    hist_eps = combined['EPS'].tolist()
                    last_label = plot_df['Label'].iloc[-1]
                    lyr, lq = int("20" + last_label.split('.')[0]), int(last_label.split('Q')[1])

                    if v3_predict_mode in ["현재 분기 예측", "다음 분기 예측"]:
                        t1q, t1y = (lq+1, lyr) if lq < 4 else (1, lyr+1)
                        per1 = current_price / (sum(hist_eps[-3:]) + est.loc['0q', 'avg'])
                        plot_df.loc[pd.Timestamp(f"{t1y}-{(t1q-1)*3+1}-01")] = [per1, np.nan, f"{str(t1y)[2:]}.Q{t1q}(E)"]

                    if v3_predict_mode == "다음 분기 예측":
                        t2q, t2y = (t1q+1, t1y) if t1q < 4 else (1, t1y+1)
                        per2 = current_price / (sum(hist_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg'])
                        plot_df.loc[pd.Timestamp(f"{t2y}-{(t2q-1)*3+1}-01")] = [per2, np.nan, f"{str(t2y)[2:]}.Q{t2q}(E)"]

                avg_per, med_per = plot_df['PER'].mean(), plot_df['PER'].median()

                # --- 그래프 크기 축소 (70% 적용) ---
                fig, ax = plt.subplots(figsize=(10.5, 4.9), facecolor='white')
                
                # 진한 유채색 (Deep Navy) 적용
                ax.plot(plot_df['Label'], plot_df['PER'], marker='o', linestyle='-', color='#003366', 
                        linewidth=2.5, markersize=7, label='Forward PER Trend', zorder=3)

                for i, label in enumerate(plot_df['Label']):
                    if "(E)" in label:
                        ax.axvspan(i-0.4, i+0.4, color='#FFCC80', alpha=0.3)
                        ax.text(i, plot_df['PER'].iloc[i] + 0.3, f"{plot_df['PER'].iloc[i]:.2f}", 
                                ha='center', fontweight='bold', color='#E65100', fontsize=8)

                ax.axhline(avg_per, color='#D32F2F', linestyle='--', linewidth=1.2, label=f'Mean: {avg_per:.2f}')
                ax.axhline(med_per, color='#7B1FA2', linestyle='-.', linewidth=1.2, label=f'Median: {med_per:.2f}')

                apply_strong_style(ax, f"[{v3_ticker}] PER Analysis (Mean vs Median)", "PER Ratio")
                
                # 범례 박스 색상을 흰색으로 변경
                ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#DDDDDD', fontsize=9, shadow=True)
                
                st.pyplot(fig)
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("현재 주가", f"${current_price:.2f}")
                c2.metric("평균 PER", f"{avg_per:.2f}x")
                c3.metric("중위 PER", f"{med_per:.2f}x")

        except Exception as e:
            st.error(f"분석 중 오류: {e}")

else:
    st.info("해당 메뉴를 선택해 주세요.")
