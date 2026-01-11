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

# --- [공통] 스타일 적용 함수 ---
def apply_strong_style(ax, title, xlabel, ylabel):
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12, color='black')
    ax.set_xlabel(xlabel, fontsize=9, fontweight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=9, fontweight='bold', color='black')
    
    # X, Y축 라인 생성 (가시성 확보)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#d3d3d3')
    ax.tick_params(axis='both', colors='black', labelsize=8)

# --- [데이터 처리 함수] ---

@st.cache_data(ttl=3600)
def fetch_valuation_data_logic_1(ticker, predict_mode):
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

        combined = pd.merge(eps_df, price_df, left_index=True, right_index=True, how='inner').sort_index()

        if predict_mode != "None":
            est = stock.earnings_estimate
            current_price = stock.fast_info['last_price'] if 'last_price' in stock.fast_info else price_df['Close'].iloc[-1]
            if est is not None and not est.empty:
                last_date_obj = pd.to_datetime(combined.index[-1])
                combined.loc[f"{(last_date_obj + pd.DateOffset(months=3)).strftime('%Y-%m')} (Est.)"] = [est['avg'].iloc[0], current_price]
                if predict_mode == "다음 분기 예측" and len(est) > 1:
                    combined.loc[f"{(last_date_obj + pd.DateOffset(months=6)).strftime('%Y-%m')} (Est.)"] = [est['avg'].iloc[1], current_price]
        return combined
    except: return None

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
    st.title(f"🚀 {main_menu}")
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            val_ticker = st.text_input("🏢 분석 티커", "TSLA").upper().strip()
        with col2:
            val_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_val = st.button("적정주가 분석 실행", type="primary", use_container_width=True)

    if run_val and val_ticker:
        combined = fetch_valuation_data_logic_1(val_ticker, val_predict_mode)
        if combined is not None:
            summary_data = []
            final_price = combined['Close'].iloc[-1]
            target_date = combined.index[-1]
            
            st.subheader(f"📊 {val_ticker} 연도별 시뮬레이션")
            for base_year in range(2017, 2026):
                df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                if len(df_plot) < 2: continue
                base_eps, base_price = df_plot.iloc[0]['EPS'], df_plot.iloc[0]['Close']
                if base_eps <= 0: continue
                scale_factor = base_price / base_eps
                df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                final_fair = df_plot.iloc[-1]['Fair_Value']
                gap_pct = ((final_price - final_fair) / final_fair) * 100
                
                summary_data.append({
                    "Base Year": base_year, "Multiplier (PER)": f"{scale_factor:.1f}x",
                    "Fair Value": f"${final_fair:.2f}", "Current Price": f"${final_price:.2f}",
                    "Gap (%)": f"{gap_pct:+.2f}%", "Status": "Overvalued" if gap_pct > 0 else "Undervalued"
                })

                fig, ax = plt.subplots(figsize=(7.5, 3.2), facecolor='white')
                ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', linewidth=2.0, marker='o', markersize=4, label='Price')
                ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', marker='s', markersize=4, label='Fair Value')
                apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Date", "Price ($)")
                ax.legend(fontsize=7, facecolor='white')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            st.subheader(f"📋 Valuation Summary (Target: {target_date})")
            st.dataframe(pd.DataFrame(summary_data), width=600, hide_index=True)
        else: st.error("데이터를 가져올 수 없습니다.")

# --- 메뉴 2: 개별종목 적정주가 분석 2 ---
elif main_menu == "개별종목 적정주가 분석 2":
    st.title("📅 발표일 기준 4분기 단위 적정주가 분석")
    with st.container(border=True):
        v2_ticker = st.text_input("🏢 분석 티커 입력", "AAPL").upper().strip()
        run_v2 = st.button("과거 기록 기반 밸류에이션 분석", type="primary", use_container_width=True)

    if run_v2 and v2_ticker:
        try:
            with st.spinner("데이터 분석 중..."):
                stock = yf.Ticker(v2_ticker)
                url = f"https://www.choicestock.co.kr/search/invest/{v2_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                raw_eps = pd.DataFrame()
                for df in dfs:
                    if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                        target_df = df.set_index(df.columns[0])
                        raw_eps = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                        raw_eps.index = pd.to_datetime(raw_eps.index, format='%y.%m.%d', errors='coerce')
                        raw_eps = raw_eps.dropna().sort_index()
                        raw_eps.columns = ['EPS']
                        break
                raw_eps = raw_eps[raw_eps.index >= "2017-01-01"]
                price_history = stock.history(start="2017-01-01", interval="1d")
                price_df = price_history['Close'].copy()
                if price_df.index.tz is not None: price_df.index = price_df.index.tz_localize(None)
                current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
                estimates = stock.earnings_estimate
                current_q_est = estimates['avg'].iloc[0] if estimates is not None else 0
                recent_3_actuals = raw_eps['EPS'].iloc[-3:].sum()
                final_target_eps = recent_3_actuals + current_q_est

                processed_data = []
                for i in range(0, len(raw_eps) - 3, 4):
                    group = raw_eps.iloc[i:i+4]
                    eps_sum = group['EPS'].sum()
                    avg_p = price_df[group.index[0]:group.index[-1]].mean()
                    is_last = (i + 4 >= len(raw_eps))
                    eps_val = final_target_eps if is_last else eps_sum
                    processed_data.append({
                        '기준 연도': f"{group.index[0].year}년", '4분기 EPS합': f"{eps_val:.2f}" + ("(예상)" if is_last else ""),
                        '평균 주가': round(avg_p, 2), '평균 PER': round(avg_p / eps_val, 1) if eps_val > 0 else 0,
                        'PER_raw': avg_p / eps_val if eps_val > 0 else 0
                    })
                avg_past_per = np.mean([d['PER_raw'] for d in processed_data if d['PER_raw'] > 0])
                st.success(f"**{v2_ticker}** 분석 완료")
                c1, c2, c3 = st.columns(3)
                c1.metric("현재 주가", f"${current_price:.2f}")
                c2.metric("현재 적정가", f"${final_target_eps * avg_past_per:.2f}")
                c3.metric("과거 평균 PER", f"{avg_past_per:.1f}x")
                
                table_list = [{'기준 연도': d['기준 연도'], '4분기 EPS합': d['4분기 EPS합'], '과거 평균주가': f"${d['평균 주가']}", '과거 PER': f"{d['평균 PER']}x", '적정 가치': f"${final_target_eps * d['PER_raw']:.2f}", '판단': "저평가" if current_price < (final_target_eps * d['PER_raw']) else "고평가"} for d in processed_data]
                st.dataframe(pd.DataFrame(table_list), width=650, hide_index=True)
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 3: 개별종목 적정주가 분석 3 (PER 추이 및 평균/중위값) ---
elif main_menu == "개별종목 적정주가 분석 3":
    st.title("📈 PER 추이 정밀 분석 (평균/중위값)")
    with st.container(border=True):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1: v3_ticker = st.text_input("🏢 분석 티커", "MSFT").upper().strip()
        with col2: v3_start_year = st.number_input("📅 시작 연도", 2010, 2025, 2017)
        with col3: v3_predict_mode = st.radio("🔮 미래 예측 포함 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_v3 = st.button("PER 정밀 분석 실행", type="primary", use_container_width=True)

    if run_v3 and v3_ticker:
        try:
            url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
            dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
            target_df = next(df.set_index(df.columns[0]) for df in dfs if 'PER' in df.iloc[:, 0].values)
            per_raw, eps_raw = target_df[target_df.index.str.contains('PER')].transpose(), target_df[target_df.index.str.contains('EPS')].transpose()
            combined = pd.DataFrame({'PER': pd.to_numeric(per_raw.iloc[:, 0], errors='coerce'), 'EPS': pd.to_numeric(eps_raw.iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')}).dropna()
            combined.index = pd.to_datetime(combined.index, format='%y.%m.%d').sort_index()

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
            fig, ax = plt.subplots(figsize=(7.5, 3.5), facecolor='white') # 요청대로 크기 축소
            ax.plot(plot_df['Label'], plot_df['PER'], marker='o', color='#0047AB', lw=1.8, ms=5, label='PER Trend')
            ax.axhline(avg_per, color='#D32F2F', ls='--', lw=1.2, label=f'Average ({avg_per:.2f})')
            ax.axhline(med_per, color='#7B1FA2', ls='-.', lw=1.2, label=f'Median ({med_per:.2f})')
            
            # 예측 구간 강조
            for i, label in enumerate(plot_df['Label']):
                if "(E)" in label: ax.axvspan(i-0.4, i+0.4, color='#FF8C00', alpha=0.1)

            apply_strong_style(ax, f"[{v3_ticker}] PER Analysis: Mean vs Median", "Quarter (Time)", "PER Value")
            ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d3d3d3', fontsize=7)
            st.pyplot(fig)
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("현재 주가", f"${current_price:.2f}"), c2.metric("평균 PER", f"{avg_per:.2f}x"), c3.metric("중위 PER", f"{med_per:.2f}x")
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 4 & 5: 기업 가치 비교 / ETF 수익률 ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.title("⚖️ 기업 가치 비교 (PER/EPS)")
    st.info("준비 중인 기능입니다.")

else:
    st.title("📊 ETF 섹터 수익률 분석")
    st.info("준비 중인 기능입니다.")
