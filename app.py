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
    
    # X, Y축 라인 생성 (가시성 확보)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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

# --- [메뉴 1: 개별종목 적정주가 분석 1] ---
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
            eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
            eps_df = eps_df.dropna()
            def adjust_date(dt):
                return (dt.replace(day=1) - timedelta(days=1)).strftime('%Y-%m') if dt.day <= 5 else dt.strftime('%Y-%m')
            eps_df.index = [adjust_date(d) for d in eps_df.index]
            eps_df['EPS'] = pd.to_numeric(eps_df['EPS'].astype(str).str.replace(',', ''), errors='coerce')
            
            stock = yf.Ticker(val_ticker)
            price_df = stock.history(start="2017-01-01", interval="1mo", auto_adjust=False)
            if price_df.index.tz is not None: price_df.index = price_df.index.tz_localize(None)
            price_df.index = price_df.index.strftime('%Y-%m')
            price_df = price_df[['Close']].copy()
            combined = pd.merge(eps_df, price_df, left_index=True, right_index=True, how='inner').sort_index()

            summary_data = []
            for base_year in range(2017, 2026):
                df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                if len(df_plot) < 2: continue
                base_eps, base_price = df_plot.iloc[0]['EPS'], df_plot.iloc[0]['Close']
                if base_eps <= 0: continue
                scale_factor = base_price / base_eps
                df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                final_price, final_fair = df_plot.iloc[-1]['Close'], df_plot.iloc[-1]['Fair_Value']
                gap_pct = ((final_price - final_fair) / final_fair) * 100
                
                summary_data.append({
                    "Base Year": base_year, "Multiplier": f"{scale_factor:.1f}x",
                    "Fair Value": f"${final_fair:.2f}", "Current Price": f"${final_price:.2f}",
                    "Gap (%)": f"{gap_pct:+.2f}%"
                })
                fig, ax = plt.subplots(figsize=(7.5, 3.2))
                ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', label='Price')
                ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', label='Fair Value')
                apply_strong_style(ax, f"Base Year: {base_year}", "Date", "Price ($)")
                st.pyplot(fig)
            st.dataframe(pd.DataFrame(summary_data), width=600)
        except Exception as e: st.error(f"오류: {e}")

# --- [메뉴 2: 개별종목 적정주가 분석 2] ---
elif main_menu == "개별종목 적정주가 분석 2":
    st.title("📅 4분기 단위 밸류에이션 기록")
    with st.container(border=True):
        v2_ticker = st.text_input("🏢 분석 티커 입력", "AAPL").upper().strip()
        run_v2 = st.button("분석 실행", type="primary", use_container_width=True)

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
            final_target_eps = raw_eps['EPS'].iloc[-3:].sum() + (stock.earnings_estimate['avg'].iloc[0] if stock.earnings_estimate is not None else 0)

            table_list = []
            for i in range(0, len(raw_eps) - 3, 4):
                group = raw_eps.iloc[i:i+4]
                eps_sum, avg_p = group['EPS'].sum(), price_df[group.index[0]:group.index[-1]].mean()
                per = avg_p / eps_sum if eps_sum > 0 else 0
                fair_p = final_target_eps * per
                table_list.append({
                    '연도': f"{group.index[0].year}년", 'EPS합': f"{eps_sum:.2f}",
                    '평균주가': f"${avg_p:.2f}", '과거PER': f"{per:.1f}x",
                    '적정가치': f"${fair_p:.2f}", '판단': "저평가" if current_price < fair_p else "고평가"
                })
            st.dataframe(pd.DataFrame(table_list), width=650, hide_index=True)
        except Exception as e: st.error(f"오류: {e}")

# --- [메뉴 3: 개별종목 적정주가 분석 3] ---
elif main_menu == "개별종목 적정주가 분석 3":
    st.title("📈 PER 추이 정밀 분석 (평균/중위값)")
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1: v3_ticker = st.text_input("🏢 분석 티커", "MSFT").upper().strip()
        with col2: v3_start_year = st.number_input("📅 시작 연도", 2010, 2025, 2017)
        with col3: v3_predict_mode = st.radio("🔮 미래 예측", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_v3 = st.button("PER 분석 실행", type="primary", use_container_width=True)

    if run_v3 and v3_ticker:
        try:
            url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
            dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
            target_df = next(df.set_index(df.columns[0]) for df in dfs if 'PER' in df.iloc[:, 0].values)
            per_raw = target_df[target_df.index.str.contains('PER')].transpose()
            eps_raw = target_df[target_df.index.str.contains('EPS')].transpose()
            combined = pd.DataFrame({'PER': pd.to_numeric(per_raw.iloc[:, 0], errors='coerce'), 
                                     'EPS': pd.to_numeric(eps_raw.iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')}).dropna()
            combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
            combined = combined.sort_index()

            def get_q_label(dt):
                y, m = (dt.year, dt.month) if dt.day > 5 else ((dt - timedelta(days=5)).year, (dt - timedelta(days=5)).month)
                return f"{str(y)[2:]}.Q{(m-1)//3 + 1}"
            combined['Label'] = [get_q_label(d) for d in combined.index]
            plot_df = combined[combined.index >= f"{v3_start_year}-01-01"].copy()

            stock = yf.Ticker(v3_ticker)
            current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
            est = stock.earnings_estimate
            
            if v3_predict_mode != "None" and est is not None:
                h_eps = combined['EPS'].tolist()
                last_l = plot_df['Label'].iloc[-1]
                ly, lq = int("20" + last_l.split('.')[0]), int(last_l.split('Q')[1])
                if v3_predict_mode in ["현재 분기 예측", "다음 분기 예측"]:
                    t1q, t1y = (lq+1, ly) if lq < 4 else (1, ly+1)
                    plot_df.loc[pd.Timestamp(f"{t1y}-{(t1q-1)*3+1}-01")] = [current_price/(sum(h_eps[-3:])+est.loc['0q','avg']), np.nan, f"{str(t1y)[2:]}.Q{t1q}(E)"]
                if v3_predict_mode == "다음 분기 예측":
                    t2q, t2y = (t1q+1, t1y) if t1q < 4 else (1, t1y+1)
                    plot_df.loc[pd.Timestamp(f"{t2y}-{(t2q-1)*3+1}-01")] = [current_price/(sum(h_eps[-2:])+est.loc['0q','avg']+est.loc['+1q','avg']), np.nan, f"{str(t2y)[2:]}.Q{t2q}(E)"]

            avg_per, med_per = plot_df['PER'].mean(), plot_df['PER'].median()
            
            # --- 그래프 수정 반영 (크기 30% 더 축소: 7.5x3.5) ---
            fig, ax = plt.subplots(figsize=(7.5, 3.5), facecolor='white')
            ax.plot(plot_df['Label'], plot_df['PER'], marker='o', color='#0047AB', lw=1.8, ms=5, label='PER Trend')
            ax.axhline(avg_per, color='#D32F2F', ls='--', lw=1.2, label=f'Average ({avg_per:.2f})')
            ax.axhline(med_per, color='#7B1FA2', ls='-.', lw=1.2, label=f'Median ({med_per:.2f})')
            
            # X/Y축 라인 및 단위 설정
            apply_strong_style(ax, f"[{v3_ticker}] PER Analysis", "Quarter (Time)", "PER Value")
            ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d3d3d3', fontsize=7)
            st.pyplot(fig)
            st.divider()
            st.metric("현재 주가", f"${current_price:.2f}"), st.metric("평균 PER", f"{avg_per:.2f}x"), st.metric("중위 PER", f"{med_per:.2f}x")
        except Exception as e: st.error(f"오류: {e}")

# --- [메뉴 4 & 5: 기업 가치 비교 / ETF 수익률] ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.title("⚖️ 기업 가치 비교")
    st.info("이곳에 기존 기업 가치 비교 로직을 입력하세요.")
else:
    st.title("📊 ETF 섹터 수익률 분석")
    st.info("이곳에 기존 ETF 분석 로직을 입력하세요.")
