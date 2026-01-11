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
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 1 ---
if main_menu == "개별종목 적정주가 분석 1":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            val_ticker = st.text_input("🏢 분석 티커", "TSLA").upper().strip()
        with col2:
            val_predict_mode = st.radio(
                "🔮 미래 예측 옵션",
                ("None", "현재 분기 예측", "다음 분기 예측"),
                horizontal=True, index=0
            )
        run_val = st.button("적정주가 분석 실행", type="primary", use_container_width=True)

    if run_val and val_ticker:
        combined = fetch_valuation_data(val_ticker, val_predict_mode)
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
                status = "Overvalued" if gap_pct > 0 else "Undervalued"
                summary_data.append({
                    "Base Year": base_year,
                    "Multiplier (PER)": f"{scale_factor:.1f}x",
                    "Fair Value": f"${final_fair_value:.2f}",
                    "Current Price": f"${final_price:.2f}",
                    "Gap (%)": f"{gap_pct:+.2f}%",
                    "Status": status
                })
                fig, ax = plt.subplots(figsize=(7.7, 3.2), facecolor='white')
                ax.text(0.02, 0.92, "● Price", color='#1f77b4', transform=ax.transAxes, fontweight='bold', fontsize=9)
                ax.text(0.12, 0.92, "■ EPS", color='#d62728', transform=ax.transAxes, fontweight='bold', fontsize=9)
                ax.plot(df_plot.index, df_plot['Close'], label='Market Price', color='#1f77b4', linewidth=2.0, marker='o', markersize=4)
                ax.plot(df_plot.index, df_plot['Fair_Value'], label=f'Fair Value (Base: {base_year})', color='#d62728', linestyle='--', marker='s', markersize=4)
                for i, idx in enumerate(df_plot.index):
                    if "(Est.)" in idx:
                        ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.15)
                apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            if summary_data:
                st.divider()
                st.subheader(f"📋 Valuation Summary (Target: {target_date})")
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=False, width=600, hide_index=True)
                st.info("💡 'Undervalued'가 많은 종목일수록 역사적 밸류에이션 하단에 있을 가능성이 높습니다.")

# --- 메뉴 2: 개별종목 적정주가 분석 2 ---
elif main_menu == "개별종목 적정주가 분석 2":
    with st.container(border=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            v2_ticker = st.text_input("🏢 분석 티커 입력", "PAYX").upper().strip()
        with col2:
            st.write("")
            st.write("")
            run_v2 = st.button("정밀 가치 분석 실행", type="primary", use_container_width=True)

    if run_v2 and v2_ticker:
        try:
            with st.spinner('데이터를 수집하고 분석 중입니다...'):
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
                price_df = stock.history(start="2017-01-01", interval="1d")['Close']
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
                    is_last_row = (i + 4 >= len(raw_eps))
                    eps_val = final_target_eps if is_last_row else eps_sum
                    processed_data.append({
                        '기준 연도': f"{group.index[0].year}년",
                        '4분기 EPS합': f"{eps_val:.2f}" + ("(예상)" if is_last_row else ""),
                        '평균 주가': f"${avg_price:.2f}",
                        '평균 PER': avg_price / eps_val if eps_val > 0 else 0
                    })

                st.subheader(f"🔍 [{v2_ticker}] 과거 밸류에이션 기록")
                display_list = []
                past_pers = [d['평균 PER'] for d in processed_data if d['평균 PER'] > 0]
                avg_past_per = np.mean(past_pers) if past_pers else 0
                for data in processed_data:
                    fair_price = final_target_eps * data['평균 PER']
                    diff_pct = ((current_price / fair_price) - 1) * 100
                    status = "🔴 고평가" if current_price > fair_price else "🔵 저평가"
                    display_list.append({
                        "기준 연도": data['기준 연도'], "4분기 EPS합": data['4분기 EPS합'], "평균 주가": data['평균 주가'],
                        "평균 PER": f"{data['평균 PER']:.1f}x", "적정주가 가치": f"${fair_price:.2f}", "현재가 판단": f"{abs(diff_pct):.1f}% {status}"
                    })
                st.dataframe(pd.DataFrame(display_list), width=750, hide_index=True)
                st.success(f"현재 실시간 주가: **${current_price:.2f}** | 과거 평균 PER기준 적정가: **${final_target_eps * avg_past_per:.2f}**")
        except: st.error("데이터 수집 중 오류가 발생했습니다.")

# --- 메뉴 3: 개별종목 적정주가 분석 3 (새로운 로직) ---
elif main_menu == "개별종목 적정주가 분석 3":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            v3_ticker = st.text_input("🏢 분석 티커 입력", "AAPL").upper().strip()
        with col2:
            v3_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        run_v3 = st.button("PER 밴드 분석 실행", type="primary", use_container_width=True)

    if run_v3 and v3_ticker:
        try:
            with st.spinner('PER 밴드 및 가치 분석 중...'):
                stock = yf.Ticker(v3_ticker)
                # 1. EPS 데이터 (TTM 환산)
                url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                raw_eps = pd.DataFrame()
                for df in dfs:
                    if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                        raw_eps = df.set_index(df.columns[0]).transpose()
                        raw_eps.index = pd.to_datetime(raw_eps.index, format='%y.%m.%d', errors='coerce')
                        raw_eps = raw_eps.dropna().sort_index()
                        raw_eps = raw_eps.iloc[:, [0]]
                        raw_eps.columns = ['EPS']
                        break
                
                # TTM EPS 계산
                raw_eps['TTM_EPS'] = raw_eps['EPS'].rolling(window=4).sum()
                raw_eps = raw_eps.dropna()

                # 2. 주가 데이터 연동
                price_df = stock.history(start=raw_eps.index[0], interval="1d")['Close']
                if price_df.index.tz is not None: price_df.index = price_df.index.tz_localize(None)

                # 매칭 데이터 생성
                valuation_df = []
                for date in raw_eps.index:
                    if date in price_df.index:
                        p = price_df.loc[date]
                        e = raw_eps.loc[date, 'TTM_EPS']
                        valuation_df.append({'Date': date, 'Price': p, 'EPS': e, 'PER': p/e if e > 0 else 0})
                
                v_df = pd.DataFrame(valuation_df).set_index('Date')
                
                # 3. PER 밴드 통계
                max_per = v_df['PER'].max()
                min_per = v_df['PER'].min()
                avg_per = v_df['PER'].mean()
                
                # 4. 미래 예측 반영
                current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
                target_eps = raw_eps['TTM_EPS'].iloc[-1]
                
                if v3_predict_mode != "None":
                    est = stock.earnings_estimate
                    if est is not None:
                        if v3_predict_mode == "현재 분기 예측":
                            target_eps = raw_eps['EPS'].iloc[-3:].sum() + est['avg'].iloc[0]
                        else: # 다음 분기
                            target_eps = raw_eps['EPS'].iloc[-2:].sum() + est['avg'].iloc[0] + est['avg'].iloc[1]

                # 5. 결과 시각화
                st.subheader(f"📈 {v3_ticker} 역사적 PER 밴드 분석")
                
                
                
                fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
                ax.plot(v_df.index, v_df['Price'], label='Price', color='black', linewidth=2)
                ax.plot(v_df.index, v_df['EPS'] * max_per, label=f'Max PER ({max_per:.1f}x)', color='red', linestyle='--', alpha=0.6)
                ax.plot(v_df.index, v_df['EPS'] * avg_per, label=f'Avg PER ({avg_per:.1f}x)', color='blue', linestyle='--', alpha=0.6)
                ax.plot(v_df.index, v_df['EPS'] * min_per, label=f'Min PER ({min_per:.1f}x)', color='green', linestyle='--', alpha=0.6)
                
                apply_strong_style(ax, f"{v3_ticker} PER Band Chart", "Price ($)")
                ax.legend(loc='upper left', fontsize=8)
                st.pyplot(fig)

                # 6. 테이블 출력
                results = {
                    "구분": ["최고 PER 기준", "평균 PER 기준", "최저 PER 기준"],
                    "적용 Multiplier": [f"{max_per:.1f}x", f"{avg_per:.1f}x", f"{min_per:.1f}x"],
                    "계산된 적정주가": [f"${target_eps*max_per:.2f}", f"${target_eps*avg_per:.2f}", f"${target_eps*min_per:.2f}"],
                    "현재가 대비 괴리율": [
                        f"{((current_price/(target_eps*max_per))-1)*100:+.1f}%",
                        f"{((current_price/(target_eps*avg_per))-1)*100:+.1f}%",
                        f"{((current_price/(target_eps*min_per))-1)*100:+.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(results), width=750, hide_index=True)
                st.info(f"💡 현재 분석에 사용된 TTM EPS: **${target_eps:.2f}** (예측 옵션: {v3_predict_mode})")
        except: st.error("분석을 수행할 수 없습니다. 티커를 확인해 주세요.")

# --- 메뉴 4 & 5: 기존 기능들 (동일하게 유지) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    # ... (기존 코드와 동일)
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1: ticker_input = st.text_input("🏢 티커 입력", "AAPL, MSFT, NVDA")
        with col2: start_year = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3: predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
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
                x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_per.index]
                for i, ticker in enumerate(indexed_per.columns):
                    ax.plot(range(len(indexed_per)), indexed_per[ticker], marker='o', label=ticker)
                apply_strong_style(ax, "Relative PER Change", "Change (%)")
                ax.set_xticks(range(len(indexed_per))); ax.set_xticklabels(x_labels, rotation=45)
                ax.legend(); st.pyplot(fig)
        else:
            all_eps = []
            for t in tickers:
                df = fetch_eps_data(t, predict_mode)
                if not df.empty: all_eps.append(df)
            if all_eps:
                full_idx = sorted(list(set().union(*(d.index for d in all_eps))))
                filtered_idx = [idx for idx in full_idx if idx >= f"{start_year}-Q1"]
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                for i, df in enumerate(all_eps):
                    t = [c for c in df.columns if c != 'type'][0]
                    plot_df = df.reindex(filtered_idx)
                    norm_vals = (plot_df[t] / plot_df[t].dropna().iloc[0] - 1) * 100
                    ax.plot(range(len(filtered_idx)), norm_vals, marker='o', label=t)
                apply_strong_style(ax, "EPS Growth", "Growth (%)")
                ax.set_xticks(range(len(filtered_idx))); ax.set_xticklabels(filtered_idx, rotation=45)
                ax.legend(); st.pyplot(fig)

else: # ETF 섹터 수익률 분석
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sector_list = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
            selected_etfs = st.multiselect("🌐 ETF 선택", sector_list, default=["SPY", "QQQ", "XLK", "XLE"])
        with col2: start_year_etf = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3: start_q_etf = st.selectbox("🔢 기준 분기", [1, 2, 3, 4], index=0)
        run_etf_btn = st.button("ETF 수익률 분석 시작", type="primary", use_container_width=True)
    if run_etf_btn and selected_etfs:
        df_etf = fetch_etf_data(selected_etfs)
        start_date = f"{start_year_etf}-{str((start_q_etf-1)*3 + 1).zfill(2)}"
        if any(df_etf.index >= start_date):
            valid_start = df_etf.index[df_etf.index >= start_date][0]
            norm_etf = (df_etf.loc[valid_start:] / df_etf.loc[valid_start:].iloc[0] - 1) * 100
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
            for ticker in norm_etf.columns:
                ax.plot(norm_etf.index, norm_etf[ticker], label=ticker)
            apply_strong_style(ax, "ETF Performance", "Return (%)")
            ax.legend(); st.pyplot(fig)
