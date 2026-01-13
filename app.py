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
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "개별종목 적정주가 분석 4", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 1 ---
if main_menu == "개별종목 적정주가 분석 1":
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        val_ticker = col1.text_input("🏢 분석 티커 입력", "TSLA").upper().strip()
        val_predict_mode = col2.radio("🔮 미래 예측 옵션 (Estimates)", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
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
                    if len(df_plot) < 2 or df_plot.iloc[0]['EPS'] <= 0: continue
                    scale_factor = df_plot.iloc[0]['Close'] / df_plot.iloc[0]['EPS']
                    df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                    last_fair_value = df_plot.iloc[-1]['Fair_Value']
                    gap_pct = ((final_price - last_fair_value) / last_fair_value) * 100
                    status = "🔴 고평가" if gap_pct > 0 else "🔵 저평가"
                    summary_list.append({"기준 연도": f"{base_year}년", "기준 PER": f"{scale_factor:.1f}x", "적정 주가": f"${last_fair_value:.2f}", "현재 주가": f"${final_price:.2f}", "괴리율 (%)": f"{gap_pct:+.1f}%", "상태": status})
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', linewidth=2.0, marker='o', markersize=4, label='Price')
                    ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', linestyle='--', marker='s', markersize=4, label='EPS')
                    for i, idx in enumerate(df_plot.index):
                        if "(Est.)" in str(idx): ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.1)
                    apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                    plt.xticks(rotation=45)
                    leg = ax.legend(loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)
                    for text in leg.get_texts():
                        if text.get_text() == 'Price': text.set_color('#1f77b4'); text.set_weight('bold')
                        elif text.get_text() == 'EPS': text.set_color('#d62728'); text.set_weight('bold')
                    st.pyplot(fig); plt.close(fig)
                if summary_list:
                    st.markdown("---"); st.subheader(f"📊 {val_ticker} 밸류에이션 종합 요약")
                    main_col, _ = st.columns([6, 4]) 
                    with main_col: st.dataframe(pd.DataFrame(summary_list), use_container_width=True, hide_index=True)
                else: st.warning("분석 가능한 흑자(EPS > 0) 데이터가 부족합니다.")
            else: st.error("데이터를 수집하지 못했습니다.")

# --- 메뉴 2: 개별종목 적정주가 분석 2 ---
elif main_menu == "개별종목 적정주가 분석 2":
    with st.container(border=True):
        col1, col2, col3 = st.columns([0.5, 0.5, 1], vertical_alignment="bottom")
        v2_ticker = col1.text_input("🏢 분석 티커 입력", "AAPL").upper().strip()
        run_v2 = col2.button("당해 EPS 기반 분석", type="primary", use_container_width=True)

    if run_v2 and v2_ticker:
        try:
            with st.spinner('데이터 수집 중...'):
                stock = yf.Ticker(v2_ticker)
                url = f"https://www.choicestock.co.kr/search/invest/{v2_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                dfs = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))
                raw_eps = pd.DataFrame()
                for df in dfs:
                    if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                        target_df = df.set_index(df.columns[0])
                        raw_eps = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                        raw_eps.index = pd.to_datetime(raw_eps.index, format='%y.%m.%d', errors='coerce')
                        raw_eps = raw_eps.dropna().sort_index(); raw_eps.columns = ['EPS']
                        break
                raw_eps = raw_eps[raw_eps.index >= "2017-01-01"]
                price_df = stock.history(start="2017-01-01")['Close']
                current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
                estimates = stock.earnings_estimate
                current_q_est = estimates['avg'].iloc[0] if estimates is not None else 0
                final_target_eps = raw_eps['EPS'].iloc[-3:].sum() + current_q_est
                processed_data = []
                for i in range(0, len(raw_eps) - 3, 4):
                    group = raw_eps.iloc[i:i+4]; eps_sum = group['EPS'].sum()
                    start_date, end_date = group.index[0], group.index[-1]
                    avg_price = price_df[start_date:end_date].mean()
                    is_last = (i + 4 >= len(raw_eps))
                    eps_disp = f"{eps_sum:.2f}" if not is_last else f"{final_target_eps:.2f}(예상)"
                    val_sum = eps_sum if not is_last else final_target_eps
                    processed_data.append({'기준 연도': f"{start_date.year}년", '4분기 EPS합': eps_disp, '평균 주가': f"${avg_price:.2f}", '평균 PER': avg_price/val_sum if val_sum>0 else 0})
                st.subheader(f"🔍 [{v2_ticker}] 과거 밸류에이션 기록")
                display_list = []
                avg_past_per = np.mean([d['평균 PER'] for d in processed_data if d['평균 PER'] > 0])
                for d in processed_data:
                    fair = final_target_eps * d['평균 PER']; diff = ((current_price/fair)-1)*100
                    display_list.append({"기준 연도": d['기준 연도'], "4분기 EPS합": d['4분기 EPS합'], "평균 주가": d['평균 주가'], "평균 PER": f"{d['평균 PER']:.1f}x", "적정주가 가치": f"${fair:.2f}", "현재가 판단": f"{abs(diff):.1f}% {'🔴 고평가' if current_price>fair else '🔵 저평가'}"})
                st.dataframe(pd.DataFrame(display_list), width=750, hide_index=True)
                cur_fair = final_target_eps * avg_past_per; cur_diff = ((current_price/cur_fair)-1)*100
                st.success(f"**[최종 요약]** 현재가 **${current_price:.2f}**는 평균 PER(**{avg_past_per:.1f}x**) 대비 **{abs(cur_diff):.1f}% {'고평가' if current_price>cur_fair else '저평가'}** 상태입니다.")
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 3: 개별종목 적정주가 분석 3 ---
elif main_menu == "개별종목 적정주가 분석 3":
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            v3_ticker = st.text_input("🏢 티커 입력", "MSFT").upper().strip()
        with col2:
            v3_start_year = st.number_input("📅 기준 연도", 2010, 2025, 2017)
        with col3:
            v3_predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
        
        v3_selected_metric = st.radio("📈 분석 지표 선택", ("PER 그래프", "PER 테이블"), horizontal=True)
        v3_analyze_btn = st.button("데이터 분석 실행", type="primary", use_container_width=True)

    if v3_analyze_btn and v3_ticker:
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
                    plot_df = combined[combined.index >= f"{v3_start_year}-01-01"].copy()

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

                    if v3_selected_metric == "PER 그래프":
                        avg_per = plot_df['PER'].mean()
                        median_per = plot_df['PER'].median()
                        max_p, min_p = plot_df['PER'].max(), plot_df['PER'].min()
                        fig, ax = plt.subplots(figsize=(12, 6.5), facecolor='white')
                        x_idx = range(len(plot_df))
                        ax.plot(x_idx, plot_df['PER'], marker='o', color='#34495e', linewidth=2.5, zorder=4, label='Forward PER')
                        ax.axhline(avg_per, color='#e74c3c', linestyle='--', linewidth=1.5, zorder=2, label=f'Average: {avg_per:.1f}')
                        ax.axhline(median_per, color='#8e44ad', linestyle='-.', linewidth=1.5, zorder=2, label=f'Median: {median_per:.1f}')
                        h_rng = max(max_p - avg_per, avg_per - min_p) * 1.6
                        ax.set_ylim(avg_per - h_rng, avg_per + h_rng)
                        leg = ax.legend(loc='upper left', frameon=True, shadow=True)
                        leg.get_frame().set_facecolor('white')
                        for text in leg.get_texts(): text.set_color('black')
                        apply_strong_style(ax, f"[{v3_ticker}] PER Valuation Trend", "PER Ratio")
                        ax.set_xticks(x_idx); ax.set_xticklabels(plot_df['Label'], rotation=45)
                        for i, (idx, row) in enumerate(plot_df.iterrows()):
                            if "(E)" in str(row['Label']):
                                ax.axvspan(i-0.4, i+0.4, color='#fff9c4', alpha=0.7)
                                ax.text(i, row['PER'] + (h_rng*0.08), f"{row['PER']:.1f}", ha='center', color='#d35400', fontweight='bold')
                        st.pyplot(fig)
                    
                    else: # PER 테이블 (수정 요청 사항 반영)
                        st.markdown(f"### <center>📊 {v3_ticker} 정밀 검증 PER 테이블</center>", unsafe_allow_html=True)
                        
                        # 데이터 피벗 (연도별/분기별 구조 생성)
                        table_data = plot_df.copy()
                        table_data['Year'] = table_data.index.year
                        table_data['Quarter'] = table_data['Label'].apply(lambda x: x.split('.')[1].replace('(E)', ''))
                        
                        # 피벗 테이블 생성
                        df_pivot = table_data.pivot(index='Year', columns='Quarter', values='PER')
                        
                        # Q1~Q4 컬럼 순서 보장 및 부족한 컬럼 채우기
                        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                            if q not in df_pivot.columns:
                                df_pivot[q] = np.nan
                        df_pivot = df_pivot[['Q1', 'Q2', 'Q3', 'Q4']].sort_index(ascending=False)

                        # 중앙 40% 배치를 위한 컬럼 설정 (3:4:3 비율)
                        left_space, mid_col, right_space = st.columns([3, 4, 3])
                        
                        with mid_col:
                            # 스타일 적용 (안팎 검정 테두리 및 배경색)
                            styled_df = df_pivot.style.format("{:.2f}", na_rep="-") \
                                .set_table_styles([
                                    # 테이블 전체 바깥 테두리
                                    {'selector': '', 'props': [('border', '2px solid black')]},
                                    # 헤더(연도, Q1~Q4) 스타일 및 테두리
                                    {'selector': 'th', 'props': [
                                        ('border', '1px solid black'), 
                                        ('background-color', '#f0f2f6'), 
                                        ('color', 'black'), 
                                        ('font-weight', 'bold'),
                                        ('text-align', 'center')
                                    ]},
                                    # 데이터 셀 안쪽 격자 테두리
                                    {'selector': 'td', 'props': [
                                        ('border', '1px solid black'), 
                                        ('text-align', 'center'),
                                        ('color', 'black')
                                    ]}
                                ])
                            
                            st.dataframe(styled_df, use_container_width=True)
                        st.info("💡 위 테이블은 연도별/분기별 PER 현황을 보여주며, 중앙 40% 너비로 최적화되었습니다.")

                else: st.warning("데이터 수집 실패")
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 4: 개별종목 적정주가 분석 4 ---
elif main_menu == "개별종목 적정주가 분석 4":
    with st.container(border=True):
        v4_ticker = st.text_input("🏢 분석 티커 입력 (PEG 분석)", "AAPL").upper().strip()
        run_v4 = st.button("연도별 정밀 PEG 분석 실행", type="primary", use_container_width=True)
    if run_v4 and v4_ticker:
        try:
            with st.spinner('데이터 수집 중...'):
                url = f"https://www.choicestock.co.kr/search/invest/{v4_ticker}/MRQ"
                dfs = pd.read_html(io.StringIO(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text))
                target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS', na=False).any()), None)
                if target_df is not None:
                    target_df = target_df.set_index(target_df.columns[0])
                    eps_df = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                    eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
                    eps_df = eps_df.dropna().sort_index(); eps_df.columns = ['Quarterly_EPS']
                    stock = yf.Ticker(v4_ticker); current_price = stock.history(period="5d")['Close'].iloc[-1]
                    try:
                        est = stock.earnings_estimate
                        curr_q_est = est['avg'].iloc[0]; next_q_est = est['avg'].iloc[1]; curr_year_est = est['avg'].iloc[2]
                    except:
                        curr_year_est = stock.info.get('forwardEps', 0); curr_q_est = curr_year_est/4; next_q_est = curr_year_est/4
                    latest_date = eps_df.index[-1]; latest_idx = len(eps_df)-1
                    def get_ttm(idx): return eps_df['Quarterly_EPS'].iloc[idx-3 : idx+1].sum() if idx >= 3 else None
                    results = []
                    current_ttm = get_ttm(latest_idx)
                    per_val = current_price / current_ttm
                    for y in range(5, 0, -1):
                        target_idx = latest_idx - (y * 4)
                        if target_idx >= 3:
                            past_ttm = get_ttm(target_idx)
                            if past_ttm > 0:
                                growth = ((current_ttm / past_ttm) ** (1/y) - 1) * 100
                                results.append({'분석 기간': f"최근 {y}년", '과거 TTM': past_ttm, '기준 TTM': current_ttm, '성장률': growth, 'PER': per_val, 'PEG': per_val/growth if growth > 0 else 0})
                    if results:
                        st.subheader(f"📌 PEG 분석 결과")
                        df_res = pd.DataFrame(results)
                        st.dataframe(df_res.style.format({'성장률': '{:.2f}%', 'PER': '{:.2f}', 'PEG': '{:.2f}'}).highlight_between(left=0.1, right=1.0, subset=['PEG'], color='#D4EDDA'), width=600, hide_index=True)
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 5: 기업 가치 비교 (PER/EPS) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1: ticker_input = st.text_input("🏢 티커 입력", "AAPL, MSFT, GOOGL")
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
                colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
                x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_per.index]
                for i, ticker in enumerate(indexed_per.columns):
                    series = indexed_per[ticker].dropna()
                    f_count = 1 if predict_mode == "현재 분기 예측" else (2 if predict_mode == "다음 분기 예측" else 0)
                    h_end = len(series) - f_count
                    ax.plot(range(h_end), series.values[:h_end], marker='o', label=f"{ticker} ({series.values[-1]:+.1f}%)", color=colors[i], linewidth=2.5)
                    if f_count > 0: ax.plot(range(h_end-1, len(series)), series.values[h_end-1:], linestyle='--', color=colors[i], linewidth=2.0, alpha=0.8)
                apply_strong_style(ax, f"Relative PER Change since {start_year}", "Change (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_xticks(range(len(indexed_per))); ax.set_xticklabels(x_labels, rotation=45)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
                st.pyplot(fig)
        else: # EPS
            all_eps = []
            for t in tickers:
                df = fetch_eps_data(t, predict_mode); 
                if not df.empty: all_eps.append(df)
            if all_eps:
                full_idx = sorted(list(set().union(*(d.index for d in all_eps))))
                filtered_idx = [idx for idx in full_idx if idx >= f"{start_year}-Q1"]
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                for i, df in enumerate(all_eps):
                    t = [c for c in df.columns if c != 'type'][0]
                    plot_df = df.reindex(filtered_idx); valid_data = plot_df[plot_df[t].notna()]
                    if valid_data.empty: continue
                    norm_vals = (plot_df[t] / valid_data[t].iloc[0] - 1) * 100
                    color = plt.cm.Set1(i % 9); act_mask = plot_df['type'] == 'Actual'
                    last_act = np.where(act_mask)[0][-1] if any(act_mask) else 0
                    ax.plot(range(last_act + 1), norm_vals.iloc[:last_act + 1], marker='o', label=f"{t} ({norm_vals.dropna().values[-1]:+.1f}%)", color=color, linewidth=2.5)
                    if predict_mode != "None": ax.plot(range(last_act, len(filtered_idx)), norm_vals.iloc[last_act:], linestyle='--', color=color, linewidth=2.0)
                apply_strong_style(ax, f"Normalized EPS Growth since {start_year}-Q1", "Growth (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_xticks(range(len(filtered_idx))); ax.set_xticklabels(filtered_idx, rotation=45)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
                st.pyplot(fig)

# --- 메뉴 6: ETF 섹터 수익률 분석 ---
else:
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        selected_etfs = col1.multiselect("🌐 ETF 선택", ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"], default=["SPY", "QQQ", "XLK"])
        start_year_etf = col2.number_input("📅 기준 연도", 2010, 2025, 2020)
        start_q_etf = col3.selectbox("🔢 기준 분기", [1, 2, 3, 4], index=0)
        run_etf_btn = st.button("ETF 수익률 분석 시작", type="primary", use_container_width=True)
    if run_etf_btn and selected_etfs:
        df_etf = fetch_etf_data(selected_etfs)
        start_date = f"{start_year_etf}-{str((start_q_etf-1)*3 + 1).zfill(2)}"
        if any(df_etf.index >= start_date):
            valid_start = df_etf.index[df_etf.index >= start_date][0]
            norm_etf = (df_etf.loc[valid_start:] / df_etf.loc[valid_start:].iloc[0] - 1) * 100
            last_vals = norm_etf.iloc[-1].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
            colors = plt.cm.get_cmap('tab10', len(selected_etfs))
            for i, ticker in enumerate(last_vals.index):
                ax.plot(norm_etf.index, norm_etf[ticker], label=f"{ticker} ({last_vals[ticker]:+.1f}%)", color=colors(i), linewidth=4.0 if ticker in ["SPY", "QQQ"] else 2.5)
            apply_strong_style(ax, f"ETF Performance since {valid_start}", "Return (%)")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ticks = [d for d in norm_etf.index if d.endswith(('-01', '-04', '-07', '-10'))]
            ax.set_xticks(ticks); ax.set_xticklabels(ticks, rotation=45)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
            st.pyplot(fig)
