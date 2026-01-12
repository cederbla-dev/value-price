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
        # [수정됨] 튜플의 맨 마지막에 '개별종목 적정주가 분석 5' 추가
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "개별종목 적정주가 분석 4", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석", "개별종목 적정주가 분석 5")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 1 ---
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

                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', 
                            linewidth=2.0, marker='o', markersize=4, label='Price')
                    ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', 
                            linestyle='--', marker='s', markersize=4, label='EPS')
                    
                    for i, idx in enumerate(df_plot.index):
                        if "(Est.)" in str(idx):
                            ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.1)

                    apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                    plt.xticks(rotation=45)
                    
                    leg = ax.legend(
                        loc='upper left', 
                        fontsize=11, 
                        frameon=True, 
                        facecolor='white', 
                        edgecolor='black', 
                        framealpha=1.0      
                    )
                    
                    for text in leg.get_texts():
                        if text.get_text() == 'Price':
                            text.set_color('#1f77b4') 
                            text.set_weight('bold')
                        elif text.get_text() == 'EPS':
                            text.set_color('#d62728') 
                            text.set_weight('bold')
                    
                    st.pyplot(fig)
                    plt.close(fig)

                if summary_list:
                    st.write("\n")
                    st.markdown("---")
                    st.subheader(f"📊 {val_ticker} 밸류에이션 종합 요약")
                    st.caption(f"분석 기준점(Target Date): {target_date_label}")

                    summary_df = pd.DataFrame(summary_list)
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

# --- 메뉴 2: 개별종목 적정주가 분석 2 ---
elif main_menu == "개별종목 적정주가 분석 2":
    with st.container(border=True):
        col1, col2, col3 = st.columns([0.5, 0.5, 1], vertical_alignment="bottom")
        with col1:
            v2_ticker = st.text_input("🏢 분석 티커 입력", "AAPL").upper().strip()
        with col2:
            run_v2 = st.button("당해 EPS 기반 분석", type="primary", use_container_width=True)
        with col3:
            pass

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
                        if raw_eps.index.tz is not None:
                            raw_eps.index = raw_eps.index.tz_localize(None)
                        break

                raw_eps = raw_eps[raw_eps.index >= "2017-01-01"]
                
                price_history = stock.history(start="2017-01-01", interval="1d")
                price_df = price_history['Close'].copy()
                if price_df.index.tz is not None:
                    price_df.index = price_df.index.tz_localize(None)
                    
                current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
                estimates = stock.earnings_estimate
                current_q_est = estimates['avg'].iloc[0] if estimates is not None else 0

                recent_3_actuals = raw_eps['EPS'].iloc[-3:].sum()
                final_target_eps = recent_3_actuals + current_q_est

                processed_data = []
                for i in range(0, len(raw_eps) - 3, 4):
                    group = raw_eps.iloc[i:i+4]
                    eps_sum = group['EPS'].sum()
                    start_date, end_date = group.index[0], group.index[-1]
                    avg_price = price_df[start_date:end_date].mean()
                    is_last_row = (i + 4 >= len(raw_eps))
                    
                    eps_display = f"{eps_sum:.2f}"
                    if is_last_row:
                        eps_display = f"{final_target_eps:.2f}(예상)"
                        eps_sum = final_target_eps
                    
                    processed_data.append({
                        '기준 연도': f"{start_date.year}년",
                        '4분기 EPS합': eps_display,
                        '평균 주가': f"${avg_price:.2f}",
                        '평균 PER': avg_price / eps_sum if eps_sum > 0 else 0,
                        'EPS_Val': eps_sum
                    })

                st.subheader(f"🔍 [{v2_ticker}] 발표일 기준 밸류에이션 기록")
                st.markdown(f"**분석 기준 EPS:** `${final_target_eps:.2f}` (최근 3개 확정 + 1개 예측)")
                
                display_list = []
                past_pers = [d['평균 PER'] for d in processed_data if d['평균 PER'] > 0]
                avg_past_per = np.mean(past_pers) if past_pers else 0

                for data in processed_data:
                    fair_price = final_target_eps * data['평균 PER']
                    diff_pct = ((current_price / fair_price) - 1) * 100
                    status = "🔴 고평가" if current_price > fair_price else "🔵 저평가"
                    
                    display_list.append({
                        "기준 연도": data['기준 연도'],
                        "4분기 EPS합": data['4분기 EPS합'],
                        "평균 주가": data['평균 주가'],
                        "평균 PER": f"{data['평균 PER']:.1f}x",
                        "적정주가 가치": f"${fair_price:.2f}",
                        "현재가 판단": f"{abs(diff_pct):.1f}% {status}"
                    })

                st.dataframe(
                    pd.DataFrame(display_list),
                    use_container_width=False,
                    width=750,
                    hide_index=True
                )

                current_fair_value = final_target_eps * avg_past_per
                current_diff = ((current_price / current_fair_value) - 1) * 100
                c_status = "고평가" if current_price > current_fair_value else "저평가"
                
                st.success(f"""
                **[최종 요약]**
                * 현재 실시간 주가: **${current_price:.2f}**
                * 과거 평균 PER(**{avg_past_per:.1f}x**) 기준 적정가: **${current_fair_value:.2f}**
                * 결과: 현재 주가는 적정가 대비 **{abs(current_diff):.1f}% {c_status}** 상태입니다.
                """)
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

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
            with st.spinner('데이터를 분석 중입니다...'):
                url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('PER|EPS').any()), None)
                
                if target_df is not None:
                    per_raw = target_df[target_df.index.astype(str).str.contains('PER')].transpose()
                    eps_raw = target_df[target_df.index.astype(str).str.contains('EPS')].transpose()
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
                    plot_df = combined[combined.index >= f"{base_year}-01-01"].copy()

                    if v3_predict_mode != "None":
                        stock = yf.Ticker(v3_ticker)
                        current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                        est = stock.earnings_estimate
                        if est is not None and not est.empty:
                            hist_eps = combined['EPS'].tolist()
                            l_lab = plot_df['Label'].iloc[-1]
                            l_yr, l_q = int("20"+l_lab.split('.')[0]), int(l_lab.split('Q')[1])
                            
                            c_q_est = est.loc['0q', 'avg']
                            t1_q, t1_yr = (l_q+1, l_yr) if l_q < 4 else (1, l_yr+1)
                            plot_df.loc[pd.Timestamp(f"{t1_yr}-{(t1_q-1)*3+1}-01")] = [current_price/(sum(hist_eps[-3:]) + c_q_est), np.nan, f"{str(t1_yr)[2:]}.Q{t1_q}(E)"]

                            if v3_predict_mode == "다음 분기 예측":
                                t2_q, t2_yr = (t1_q+1, t1_yr) if t1_q < 4 else (1, t1_yr+1)
                                plot_df.loc[pd.Timestamp(f"{t2_yr}-{(t2_q-1)*3+1}-01")] = [current_price/(sum(hist_eps[-2:]) + c_q_est + est.loc['+1q', 'avg']), np.nan, f"{str(t2_yr)[2:]}.Q{t2_q}(E)"]

                    avg_per = plot_df['PER'].mean()
                    median_per = plot_df['PER'].median()
                    max_p, min_p = plot_df['PER'].max(), plot_df['PER'].min()
                    
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='white')
                    ax.set_facecolor('white')
                    
                    x_idx = range(len(plot_df))
                    ax.plot(x_idx, plot_df['PER'], marker='o', color='#34495e', linewidth=2.5, zorder=4, label='Forward PER')
                    ax.axhline(avg_per, color='#e74c3c', linestyle='--', linewidth=1.5, zorder=2, label=f'Average: {avg_per:.1f}')
                    ax.axhline(median_per, color='#8e44ad', linestyle='-.', linewidth=1.5, zorder=2, label=f'Median: {median_per:.1f}')
                    
                    h_rng = max(max_p - avg_per, avg_per - min_p) * 1.6
                    ax.set_ylim(avg_per - h_rng, avg_per + h_rng)

                    leg = ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
                    leg.get_frame().set_facecolor('white')
                    leg.get_frame().set_edgecolor('black')
                    for text in leg.get_texts():
                        text.set_color('black')

                    x_pos = len(plot_df) - 0.5
                    ax.text(x_pos, avg_per, f' Average: {avg_per:.1f}', color='#e74c3c', va='center', fontweight='bold', fontsize=9)
                    ax.text(x_pos, median_per, f' Median: {median_per:.1f}', color='#8e44ad', va='center', fontweight='bold', fontsize=9)

                    ax.set_title(f"[{v3_ticker}] PER Valuation Trend (Mean vs Median)", fontsize=15, pad=25, color='black', fontweight='bold')
                    ax.set_ylabel("PER Ratio", fontsize=11, color='black', fontweight='bold')
                    ax.set_xlabel("Quarter (Time)", fontsize=11, color='black', fontweight='bold')
                    ax.set_xticks(x_idx)
                    ax.set_xticklabels(plot_df['Label'], rotation=45, fontsize=10, color='black')
                    
                    ax.grid(True, axis='y', linestyle=':', alpha=0.5, color='gray')
                    for s in ax.spines.values():
                        s.set_visible(True)
                        s.set_edgecolor('black')

                    for i, (idx, row) in enumerate(plot_df.iterrows()):
                        if "(E)" in str(row['Label']):
                            ax.axvspan(i-0.4, i+0.4, color='#fff9c4', alpha=0.7, zorder=1)
                            ax.text(i, row['PER'] + (h_rng*0.08), f"{row['PER']:.1f}", ha='center', color='#d35400', fontweight='bold')

                    plt.tight_layout()
                    st.pyplot(fig)
                    
                else: st.warning("데이터 수집 실패")
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 4: 개별종목 적정주가 분석 4 ---
elif main_menu == "개별종목 적정주가 분석 4":
    with st.container(border=True):
        v4_ticker = st.text_input("🏢 분석 티커 입력 (PEG 분석)", "AAPL").upper().strip()
        run_v4 = st.button("연도별 정밀 PEG 분석 실행", type="primary", use_container_width=True)

    if run_v4 and v4_ticker:
        try:
            with st.spinner(f"[{v4_ticker}] 데이터 수집 및 연도별 정밀 분석 중..."):
                url = f"https://www.choicestock.co.kr/search/invest/{v4_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                
                resp = requests.get(url, headers=headers, timeout=10)
                dfs = pd.read_html(io.StringIO(resp.text))
                
                target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS', na=False).any()), None)
                
                if target_df is None:
                    st.error("⚠️ 해당 종목의 분기별 EPS 데이터를 찾을 수 없습니다.")
                else:
                    target_df = target_df.set_index(target_df.columns[0])
                    eps_df = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                    eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
                    eps_df = eps_df.dropna().sort_index()
                    eps_df.columns = ['Quarterly_EPS']
                    
                    stock = yf.Ticker(v4_ticker)
                    hist = stock.history(period="5d")
                    if hist.empty:
                        st.error("⚠️ 주가 데이터를 가져올 수 없습니다. 티커를 확인하세요.")
                        st.stop()
                    
                    current_price = hist['Close'].iloc[-1]
                    
                    try:
                        estimates = stock.earnings_estimate
                        if estimates is None or estimates.empty:
                            curr_year_est = stock.info.get('forwardEps', 0)
                            curr_q_est = curr_year_est / 4
                        else:
                            curr_q_est = estimates['avg'].iloc[0]
                    except:
                        curr_year_est = stock.info.get('forwardEps', 0)
                        curr_q_est = curr_year_est / 4

                    latest_date = eps_df.index[-1]
                    latest_month = latest_date.month
                    latest_idx = len(eps_df) - 1

                    def get_ttm(idx):
                        if idx < 3: return None
                        return eps_df['Quarterly_EPS'].iloc[idx-3 : idx+1].sum()

                    results = []
                    
                    # [끊긴 부분 보완 및 로직 마무리]
                    # 단순화를 위해 최신 TTM 기준 계산으로 로직 완성
                    current_ttm = get_ttm(latest_idx)
                    if current_ttm and current_ttm > 0:
                        per_val = current_price / current_ttm
                        for y in range(5, 0, -1):
                            target_idx = latest_idx - (y * 4)
                            if target_idx >= 3:
                                past_ttm = get_ttm(target_idx)
                                if past_ttm and past_ttm > 0:
                                    # CAGR Growth 계산
                                    growth = ((current_ttm / past_ttm) ** (1/y) - 1) * 100
                                    peg = per_val / growth if growth > 0 else 0
                                    results.append({
                                        '분석 기간': f"최근 {y}년",
                                        '과거 TTM': past_ttm,
                                        '현재 TTM': current_ttm,
                                        '성장률(%)': round(growth, 2),
                                        'PER': round(per_val, 2),
                                        'PEG': round(peg, 2)
                                    })
                    
                    if results:
                        st.subheader(f"📊 {v4_ticker} 연도별 PEG 분석 결과")
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                    else:
                        st.warning("분석에 필요한 과거 데이터가 충분하지 않거나 적자 기업입니다.")

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

# --- (본문 코드가 없으므로 메뉴 선택 시 에러 방지용) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    st.info("이 메뉴의 기능은 현재 코드에 포함되어 있지 않습니다.")
elif main_menu == "ETF 섹터 수익률 분석":
    st.info("이 메뉴의 기능은 현재 코드에 포함되어 있지 않습니다.")

# --- [추가된 메뉴] 개별종목 적정주가 분석 5 ---
elif main_menu == "개별종목 적정주가 분석 5":
    st.header("🔍 개별종목 적정주가 분석 5")
    st.warning("🚧 준비 중인 메뉴입니다. 본문 내용은 추후 업데이트될 예정입니다.")
