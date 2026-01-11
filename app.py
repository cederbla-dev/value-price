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
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
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
                base_eps = df_plot.iloc[0]['EPS']
                base_price = df_plot.iloc[0]['Close']
                if base_eps <= 0: continue

                scale_factor = base_price / base_eps
                df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                final_fair_value = df_plot.iloc[-1]['Fair_Value']
                gap_pct = ((final_price - final_fair_value) / final_fair_value) * 100
                
                summary_data.append({
                    "Base Year": base_year, "Multiplier (PER)": f"{scale_factor:.1f}x",
                    "Fair Value": f"${final_fair_value:.2f}", "Current Price": f"${final_price:.2f}",
                    "Gap (%)": f"{gap_pct:+.2f}%", "Status": "Overvalued" if gap_pct > 0 else "Undervalued"
                })

                fig, ax = plt.subplots(figsize=(7.7, 3.2), facecolor='white')
                ax.text(0.02, 0.92, "● Price", color='#1f77b4', transform=ax.transAxes, fontweight='bold', fontsize=9)
                ax.text(0.12, 0.92, "■ EPS", color='#d62728', transform=ax.transAxes, fontweight='bold', fontsize=9)
                ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', linewidth=2.0, marker='o', markersize=4)
                ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', marker='s', markersize=4)
                apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            if summary_data:
                st.divider()
                st.subheader(f"📋 Valuation Summary (Target: {target_date})")
                st.dataframe(pd.DataFrame(summary_data), use_container_width=False, width=600, hide_index=True)
        else:
            st.error("데이터를 가져올 수 없습니다.")

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
                    avg_price = price_df[group.index[0]:group.index[-1]].mean()
                    year_label = f"{group.index[0].year}년"
                    is_last_row = (i + 4 >= len(raw_eps))
                    display_eps = eps_sum
                    if is_last_row:
                        eps_sum = final_target_eps
                        display_eps = f"{eps_sum:.2f}(예상)"
                    
                    processed_data.append({
                        '기준 연도': year_label,
                        '4분기 EPS합': display_eps,
                        '평균 주가': round(avg_price, 2),
                        '평균 PER': round(avg_price / eps_sum, 1) if eps_sum > 0 else 0,
                        'EPS_raw': eps_sum,
                        'PER_raw': avg_price / eps_sum if eps_sum > 0 else 0
                    })

                past_pers = [d['PER_raw'] for d in processed_data if d['PER_raw'] > 0]
                avg_past_per = np.mean(past_pers) if past_pers else 0
                current_fair_value = final_target_eps * avg_past_per
                current_diff = ((current_price / current_fair_value) - 1) * 100
                current_status = "저평가" if current_price < current_fair_value else "고평가"

                st.success(f"**{v2_ticker}** 분석 완료")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("현재 주가", f"${current_price:.2f}")
                c2.metric("현재 적정가", f"${current_fair_value:.2f}", f"{current_diff:.1f}% {current_status}", delta_color="inverse")
                c3.metric("과거 평균 PER", f"{avg_past_per:.1f}x")

                st.subheader("📋 과거 4분기 단위 밸류에이션 기록")
                table_list = []
                for data in processed_data:
                    fair_price = final_target_eps * data['PER_raw']
                    diff_pct = ((current_price / fair_price) - 1) * 100
                    status = "저평가" if current_price < fair_price else "고평가"
                    
                    table_list.append({
                        '기준 연도': data['기준 연도'],
                        '4분기 EPS합': data['4분기 EPS합'],
                        '과거 평균주가': f"${data['평균 주가']}",
                        '과거 PER': f"{data['평균 PER']}x",
                        '적정 가치': f"${fair_price:.2f}",
                        '판단': f"{abs(diff_pct):.1f}% {status}"
                    })
                
                # --- 가로 길이 축소 적용 (width=650) ---
                st.dataframe(pd.DataFrame(table_list), use_container_width=False, width=650, hide_index=True)
                st.info(f"💡 **계산 근거**: 최근 확정분 합({recent_3_actuals:.2f}) + 야후 분석가 예측({current_q_est:.2f}) = **Target EPS {final_target_eps:.2f}**")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# (기업 가치 비교 및 ETF 섹터 분석 메뉴는 기존과 동일하게 유지)
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.info("메뉴를 선택해 주세요.") 

else:
    st.info("메뉴를 선택해 주세요.")
