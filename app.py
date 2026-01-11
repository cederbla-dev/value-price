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
st.set_page_config(page_title="Stock Professional Analyzer", layout="wide")

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
    """서로 다른 분기 마감일을 표준 분기(3, 6, 9, 12월)로 조정"""
    month = dt.month
    year = dt.year
    if month in [1, 2, 3]:   new_month, new_year = 3, year
    elif month in [4, 5, 6]: new_month, new_year = 6, year
    elif month in [7, 8, 9]: new_month, new_year = 9, year
    elif month in [10, 11, 12]: new_month, new_year = 12, year
    return pd.Timestamp(year=new_year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

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
        # 데이터 수집 (Logic 1)
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
            price_df = price_df[~price_df.index.duplicated(keep='last')]
            combined = pd.merge(eps_df, price_df, left_index=True, right_index=True, how='inner').sort_index()

            # 미래 예측 추가
            if val_predict_mode != "None":
                est = stock.earnings_estimate
                current_price = stock.fast_info.get('last_price', price_df['Close'].iloc[-1])
                if est is not None and not est.empty:
                    last_date_obj = pd.to_datetime(combined.index[-1])
                    combined.loc[f"{(last_date_obj + pd.DateOffset(months=3)).strftime('%Y-%m')} (Est.)"] = [est['avg'].iloc[0], current_price]
                    if val_predict_mode == "다음 분기 예측" and len(est) > 1:
                        combined.loc[f"{(last_date_obj + pd.DateOffset(months=6)).strftime('%Y-%m')} (Est.)"] = [est['avg'].iloc[1], current_price]

            summary_data = []
            st.subheader(f"📊 {val_ticker} 연도별 시뮬레이션")
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
                    "Base Year": base_year, "Multiplier (PER)": f"{scale_factor:.1f}x",
                    "Fair Value": f"${final_fair:.2f}", "Current Price": f"${final_price:.2f}",
                    "Gap (%)": f"{gap_pct:+.2f}%", "Status": "Overvalued" if gap_pct > 0 else "Undervalued"
                })

                fig, ax = plt.subplots(figsize=(7.7, 3.2), facecolor='white') # 80% 축소
                ax.text(0.02, 0.92, "● Price", color='#1f77b4', transform=ax.transAxes, fontweight='bold', fontsize=9)
                ax.text(0.12, 0.92, "■ EPS", color='#d62728', transform=ax.transAxes, fontweight='bold', fontsize=9)
                ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', linewidth=2.0, marker='o', markersize=4)
                ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', marker='s', markersize=4)
                apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            st.subheader("📋 Valuation Summary")
            st.dataframe(pd.DataFrame(summary_data), use_container_width=False, width=600, hide_index=True)
        except: st.error("데이터 수집 중 오류가 발생했습니다.")

# --- [메뉴 2: 개별종목 적정주가 분석 2] ---
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

            processed_data, table_list = [], []
            for i in range(0, len(raw_eps) - 3, 4):
                group = raw_eps.iloc[i:i+4]
                eps_sum, avg_price = group['EPS'].sum(), price_df[group.index[0]:group.index[-1]].mean()
                is_last = (i + 4 >= len(raw_eps))
                if is_last: eps_sum = final_target_eps
                
                per = avg_price / eps_sum if eps_sum > 0 else 0
                processed_data.append({'PER_raw': per})
                
                fair_price = final_target_eps * per
                diff_pct = ((current_price / fair_price) - 1) * 100
                table_list.append({
                    '기준 연도': f"{group.index[0].year}년", '4분기 EPS합': f"{eps_sum:.2f}" + ("(예상)" if is_last else ""),
                    '과거 평균주가': f"${avg_price:.2f}", '과거 PER': f"{per:.1f}x",
                    '적정 가치': f"${fair_price:.2f}", '판단': f"{abs(diff_pct):.1f}% " + ("저평가" if current_price < fair_price else "고평가")
                })
            
            avg_past_per = np.mean([d['PER_raw'] for d in processed_data if d['PER_raw'] > 0])
            st.success(f"{v2_ticker} 분석 완료")
            c1, c2, c3 = st.columns(3)
            c1.metric("현재 주가", f"${current_price:.2f}")
            c2.metric("현재 적정가", f"${final_target_eps * avg_past_per:.2f}")
            c3.metric("과거 평균 PER", f"{avg_past_per:.1f}x")
            
            st.subheader("📋 과거 4분기 단위 밸류에이션 기록")
            st.dataframe(pd.DataFrame(table_list), use_container_width=False, width=650, hide_index=True)
        except: st.error("분석 중 오류 발생")

# --- [메뉴 3: 개별종목 적정주가 분석 3] ---
elif main_menu == "개별종목 적정주가 분석 3":
    st.title("🔄 회계 주기 동기화 PER 추이 비교")
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            v3_tickers = st.text_input("🏢 비교 종목 (예: AAPL, AVGO, NKE)", "AAPL, AVGO, NKE").upper().replace(',', ' ').split()
        with col2:
            v3_start_year = st.number_input("📅 기준 연도", 2010, 2025, 2017)
        with col3:
            v3_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_v3 = st.button("동기화 분석 실행", type="primary", use_container_width=True)

    if run_v3 and v3_tickers:
        master_df = pd.DataFrame()
        for t in v3_tickers:
            try:
                url = f"https://www.choicestock.co.kr/search/invest/{t}/MRQ"
                dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
                target_df = next(df.set_index(df.columns[0]) for df in dfs if 'PER' in df.iloc[:, 0].values)
                combined = pd.DataFrame({
                    'PER': pd.to_numeric(target_df[target_df.index.str.contains('PER')].transpose().iloc[:, 0], errors='coerce'),
                    'EPS': pd.to_numeric(target_df[target_df.index.str.contains('EPS')].transpose().iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
                }).dropna()
                combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
                
                if v3_predict_mode != "None":
                    s = yf.Ticker(t)
                    cur_p = s.fast_info.get('last_price', s.history(period="1d")['Close'].iloc[-1])
                    est = s.earnings_estimate
                    if est is not None:
                        q1_dt = combined.index[-1] + pd.DateOffset(months=3)
                        combined.loc[q1_dt, 'PER'] = cur_p / (combined['EPS'].iloc[-3:].sum() + est.loc['0q', 'avg'])
                        if v3_predict_mode == "다음 분기 예측":
                            combined.loc[q1_dt + pd.DateOffset(months=3), 'PER'] = cur_p / (combined['EPS'].iloc[-2:].sum() + est.loc['0q', 'avg'] + est.loc['+1q', 'avg'])
                
                combined.index = combined.index.map(normalize_to_standard_quarter)
                master_df[t] = combined[~combined.index.duplicated(keep='last')]['PER']
            except: continue
        
        if not master_df.empty:
            master_df = master_df[master_df.index >= f"{v3_start_year}-01-01"].sort_index()
            indexed_df = (master_df / master_df.apply(lambda x: x.dropna().iloc[0])) * 100
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            for t in indexed_df.columns:
                series = indexed_df[t].dropna()
                f_count = 1 if v3_predict_mode == "현재 분기 예측" else (2 if v3_predict_mode == "다음 분기 예측" else 0)
                ax.plot(range(len(series)-f_count), series.values[:-f_count] if f_count>0 else series.values, marker='o', label=f"{t} ({series.iloc[-1]:.1f})")
                if f_count > 0:
                    ax.plot(range(len(series)-f_count-1, len(series)), series.values[-f_count-1:], linestyle='--', alpha=0.7)
            apply_strong_style(ax, f"Synced PER Index (Base 100 at {v3_start_year})", "Relative Index")
            ax.set_xticks(range(len(indexed_df))); ax.set_xticklabels([f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_df.index], rotation=45)
            ax.legend(); st.pyplot(fig)

# --- [기타 메뉴: 간단 안내] ---
else:
    st.info("해당 메뉴의 상세 로직을 구현 중이거나 이전 코드 섹션을 참조하세요.")
