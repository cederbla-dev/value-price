# app-v6-Test1.py (수정본)
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

# 추가 import for interactive grid
# pip install st-aggrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

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

# --- 추가 유틸 함수: Choicestock / Yahoo recent EPS 및 Estimates 파싱 ---
def fetch_recent_eps_from_choicestock(ticker, n_actual=4):
    """
    Choicestock에서 분기별 EPS를 가져와서 최신 n_actual개의 '실적' EPS 값을 리스트로 반환.
    (최신부터 역순으로)
    """
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(resp.text), flavor='lxml')
        target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS').any()), None)
        if target_df is None:
            return []
        target_df = target_df.set_index(target_df.columns[0])
        eps_df = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
        eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
        eps_df = eps_df.dropna().sort_index()
        eps_df.columns = ['EPS']
        vals = pd.to_numeric(eps_df['EPS'].astype(str).str.replace(',', ''), errors='coerce').dropna().tolist()
        # vals: 오래된->최신, 뒤집어 최신->오래된
        vals = vals[::-1]
        return vals[:n_actual]
    except Exception:
        return []

def fetch_yahoo_analysis_avg_estimates(ticker):
    """
    Yahoo finance analysis 페이지에서 Earnings Estimate 테이블의
    'Avg. Estimate' 행에서 'Current Qtr.' 및 'Next Qtr.' 칼럼값을 반환.
    반환: (current_q_est, next_q_est) — 숫자(소수) 또는 (None, None)
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/analysis?p={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(io.StringIO(r.text))
        for t in tables:
            try:
                # 문자열 포함여부 확인
                if any("Avg. Estimate" in str(x) for x in t.iloc[:,0].astype(str).values):
                    t2 = t.set_index(t.columns[0])
                    cols = list(t2.columns)
                    cur_col = next((c for c in cols if 'Current' in str(c) and 'Qtr' in str(c)), None)
                    nxt_col = next((c for c in cols if 'Next' in str(c) and 'Qtr' in str(c)), None)
                    if cur_col is None and len(cols) >= 1: cur_col = cols[0]
                    if nxt_col is None and len(cols) >= 2: nxt_col = cols[1]
                    cur_val = t2.loc['Avg. Estimate', cur_col] if 'Avg. Estimate' in t2.index else None
                    nxt_val = t2.loc['Avg. Estimate', nxt_col] if 'Avg. Estimate' in t2.index else None
                    def to_num(x):
                        if pd.isna(x): return None
                        try:
                            return float(str(x).replace(',',''))
                        except:
                            try:
                                return float(str(x).replace('−','-'))
                            except:
                                return None
                    return to_num(cur_val), to_num(nxt_val)
            except Exception:
                continue
        return None, None
    except Exception:
        return None, None

def compute_recent_4q_eps_sum(ticker, predict_mode):
    """
    사용자의 요구사항에 따른 최근 4분기 EPS 합 계산 함수.
    predict_mode in ("None", "현재 분기 예측", "다음 분기 예측")
    """
    recent_actuals = fetch_recent_eps_from_choicestock(ticker, n_actual=4)  # 최신부터
    if predict_mode == "None":
        if len(recent_actuals) >= 4:
            return sum(recent_actuals[:4])
        else:
            return sum(recent_actuals)
    if predict_mode == "현재 분기 예측":
        cur_est, _ = fetch_yahoo_analysis_avg_estimates(ticker)
        if cur_est is None:
            stock = yf.Ticker(ticker)
            try:
                est = stock.earnings_estimate
                cur_est = est['avg'].iloc[0] if est is not None and not est.empty else None
            except:
                cur_est = None
        sum3 = sum(recent_actuals[:3]) if len(recent_actuals) >= 3 else sum(recent_actuals)
        return sum3 + (cur_est if cur_est is not None else 0)
    if predict_mode == "다음 분기 예측":
        cur_est, next_est = fetch_yahoo_analysis_avg_estimates(ticker)
        if cur_est is None or next_est is None:
            stock = yf.Ticker(ticker)
            try:
                est = stock.earnings_estimate
                if est is not None and not est.empty:
                    if cur_est is None:
                        cur_est = est['avg'].iloc[0] if len(est) > 0 else None
                    if next_est is None and len(est) > 1:
                        next_est = est['avg'].iloc[1]
            except:
                pass
        sum2 = sum(recent_actuals[:2]) if len(recent_actuals) >= 2 else sum(recent_actuals)
        return sum2 + (cur_est if cur_est is not None else 0) + (next_est if next_est is not None else 0)

# --- [UI 레이아웃] ---

with st.sidebar:
    st.title("📂 분석 메뉴")
    main_menu = st.radio(
        "분석 종류를 선택하세요:",
        ("개별종목 적정주가 분석 1", "개별종목 적정주가 분석 2", "개별종목 적정주가 분석 3", "개별종목 적정주가 분석 4", "기업 가치 비교 (PER/EPS)", "ETF 섹터 수익률 분석")
    )

st.title(f"🚀 {main_menu}")

# --- 메뉴 1: 개별종목 적정주가 분석 1 (범례 배경색 및 정렬 최종 수정본) ---
if main_menu == "개별종목 적정주가 분석 1":
    # (기존 코드 유지, 생략하지 않고 파일에는 있음)
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

                    fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
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
                        bbox_to_anchor=(1.02, 1),
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

                st.subheader(f"🔍 [{v2_ticker}] 발표일 기준 과거 밸류에이션 기록")
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

# --- 메뉴 3: 개별종목 적정주가 분석 3 (수정됨: PER 테이블에 선택기능 추가) ---
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

                        fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                        x_idx = range(len(plot_df))
                        ax.plot(x_idx, plot_df['PER'], marker='o', color='#34495e', linewidth=2.5, zorder=4, label='Forward PER')
                        ax.axhline(avg_per, color='#e74c3c', linestyle='--', linewidth=1.5, zorder=2, label=f'Average: {avg_per:.1f}')
                        ax.axhline(median_per, color='#8e44ad', linestyle='-.', linewidth=1.5, zorder=2, label=f'Median: {median_per:.1f}')
                        
                        h_rng = max(max_p - avg_per, avg_per - min_p) * 1.6
                        ax.set_ylim(avg_per - h_rng, avg_per + h_rng)

                        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, shadow=True)
                        leg.get_frame().set_facecolor('white')
                        for text in leg.get_texts(): text.set_color('black')

                        apply_strong_style(ax, f"[{v3_ticker}] PER Valuation Trend", "PER Ratio")
                        ax.set_xticks(x_idx); ax.set_xticklabels(plot_df['Label'], rotation=45)
                        
                        for i, (idx, row) in enumerate(plot_df.iterrows()):
                            if "(E)" in str(row['Label']):
                                ax.axvspan(i-0.4, i+0.4, color='#fff9c4', alpha=0.7)
                                ax.text(i, row['PER'] + (h_rng*0.08), f"{row['PER']:.1f}", ha='center', color='#d35400', fontweight='bold')
                        
                        st.pyplot(fig)
                    
                    else: # PER 테이블 -> 여기에 상호작용 기능 추가 (st-aggrid)
                        st.subheader(f"📊 {v3_ticker} 연도별/분기별 PER 데이터 리스트 (인터랙티브 선택 가능)")
                        table_df = plot_df[['Label', 'PER']].copy()
                        table_df['Year'] = table_df['Label'].apply(lambda x: "20" + x.split('.')[0])
                        table_df['Quarter'] = table_df['Label'].apply(lambda x: x.split('.')[1].replace('(E)', ''))
                        pivot_df = table_df.pivot(index='Year', columns='Quarter', values='PER')
                        pivot_df = pivot_df.reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'])
                        pivot_df.index.name = 'Year'
                        pivot_df = pivot_df.reset_index()

                        display_df = pivot_df.copy()

                        # AgGrid 셋업
                        gb = GridOptionsBuilder.from_dataframe(display_df)
                        gb.configure_default_column(editable=False, resizable=True, filter=True, sortable=True)
                        grid_options = gb.build()
                        # enable cell range selection
                        grid_options['enableRangeSelection'] = True
                        grid_options['suppressMultiRangeSelection'] = False

                        st.markdown("**사용법**: 표에서 마우스 드래그(또는 Ctrl+클릭)로 PER 셀들을 선택하세요. 그 다음 버튼을 눌러 계산합니다.")
                        # 그리드 표시
                        grid_response = AgGrid(
                            display_df,
                            gridOptions=grid_options,
                            allow_unsafe_jscode=True,
                            enable_enterprise_modules=False,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                            fit_columns_on_grid_load=False,
                            height=320
                        )

                        # 선택값 추출
                        selected_values = []
                        if isinstance(grid_response, dict):
                            sel_cells = grid_response.get('selected_cells') or grid_response.get('cellRanges') or grid_response.get('range_selection') or None
                            if sel_cells:
                                for c in sel_cells:
                                    if isinstance(c, dict) and 'value' in c:
                                        selected_values.append(c['value'])
                                    elif isinstance(c, dict) and 'cells' in c:
                                        for cc in c['cells']:
                                            if 'value' in cc:
                                                selected_values.append(cc['value'])
                            else:
                                sel_rows = grid_response.get('selected_rows', [])
                                if sel_rows:
                                    for r in sel_rows:
                                        for q in ['Q1','Q2','Q3','Q4']:
                                            v = r.get(q)
                                            if pd.notna(v):
                                                selected_values.append(v)

                        # 버튼: 1) PER 선택 안내(UX), 2) 평균 구하기, 3) 적정주가 구하기
                        col_a, col_b, col_c = st.columns([1,1,1])
                        with col_a:
                            st.button("① PER 선택 (표에서 셀을 선택하세요)", key=f"select_btn_{v3_ticker}")
                        with col_b:
                            per_mean_btn = st.button("② 선택 PER 평균 구하기", key=f"mean_btn_{v3_ticker}")
                        with col_c:
                            fair_price_btn = st.button("③ 적정주가 구하기", key=f"fair_btn_{v3_ticker}")

                        if 'selected_per_values' not in st.session_state:
                            st.session_state['selected_per_values'] = []

                        if per_mean_btn:
                            numeric_vals = []
                            for v in selected_values:
                                try:
                                    if v is None or (isinstance(v, float) and np.isnan(v)):
                                        continue
                                    sval = str(v).replace('x','').replace(',','').strip()
                                    numeric_vals.append(float(sval))
                                except:
                                    continue
                            if not numeric_vals:
                                st.warning("선택된 PER 값이 없습니다. 표에서 값을 선택해 주세요.")
                            else:
                                st.session_state['selected_per_values'] = numeric_vals
                                mean_per = float(np.mean(numeric_vals))
                                st.success(f"선택된 PER 개수: {len(numeric_vals)} → 평균 PER = {mean_per:.2f}x")
                                st.write(numeric_vals)

                        if fair_price_btn:
                            numeric_vals = st.session_state.get('selected_per_values', [])
                            if not numeric_vals:
                                tmp_numeric = []
                                for v in selected_values:
                                    try:
                                        if v is None or (isinstance(v, float) and np.isnan(v)):
                                            continue
                                        tmp_numeric.append(float(str(v).replace('x','').replace(',','').strip()))
                                    except:
                                        continue
                                numeric_vals = tmp_numeric

                            if not numeric_vals:
                                st.warning("평균 PER을 구할 수 없습니다. 먼저 셀을 선택한 뒤 '선택 PER 평균 구하기'를 눌러주세요.")
                            else:
                                mean_per = float(np.mean(numeric_vals))
                                eps_sum = compute_recent_4q_eps_sum(v3_ticker, v3_predict_mode)
                                if eps_sum is None:
                                    st.error("EPS 합계를 계산할 수 없습니다. 데이터 소스(Choicestock / Yahoo)가 응답하는지 확인하세요.")
                                else:
                                    fair_price = mean_per * eps_sum
                                    st.success(f"📌 적정주가 = 평균 PER({mean_per:.2f}x) × 최근 4분기 EPS 합({eps_sum:.2f}) = ${fair_price:.2f}")
                                    st.caption(f"계산 방식: predict_mode = {v3_predict_mode}")
                else:
                    st.warning("데이터 수집 실패")
        except Exception as e:
            st.error(f"오류: {e}")

# --- 메뉴 4: 개별종목 적정주가 분석 4 ---
elif main_menu == "개별종목 적정주가 분석 4":
    with st.container(border=True):
        v4_ticker = st.text_input("🏢 분석 티커 입력 (PEG 분석)", "AAPL").upper().strip()
        run_v4 = st.button("연도별 정밀 PEG 분석 실행", type="primary", use_container_width=True)

    if run_v4 and v4_ticker:
        try:
            with st.spinner(f"[{v4_ticker}] 데이터 수집 및 연도별 정밀 분석 중..."):
                url = f"https://www.choicestock.co.kr/search/invest/{v4_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                
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
                            next_q_est = curr_year_est / 4
                        else:
                            curr_q_est = estimates['avg'].iloc[0]
                            next_q_est = estimates['avg'].iloc[1]
                            curr_year_est = estimates['avg'].iloc[2]
                    except:
                        curr_year_est = stock.info.get('forwardEps', 0)
                        curr_q_est = curr_year_est / 4
                        next_q_est = curr_year_est / 4

                    latest_date = eps_df.index[-1]
                    latest_month = latest_date.month
                    latest_idx = len(eps_df) - 1

                    def get_ttm(idx):
                        if idx < 3: return None
                        return eps_df['Quarterly_EPS'].iloc[idx-3 : idx+1].sum()

                    results = []
                    analysis_type = ""
                    base_date = latest_date

                    if latest_month in [10, 11, 12]:
                        analysis_type = "[확정 실적 기준] 연간 PEG 요약"
                        current_ttm = get_ttm(latest_idx)
                        per_val = current_price / current_ttm
                        for y in range(5, 0, -1):
                            target_idx = latest_idx - (y * 4)
                            if target_idx >= 3:
                                past_ttm = get_ttm(target_idx)
                                if past_ttm > 0:
                                    growth = ((current_ttm / past_ttm) ** (1/y) - 1) * 100
                                    results.append({'분석 기간': f"최근 {y}년 연간", '과거 TTM': past_ttm, '기준 TTM': current_ttm, '성장률': growth, 'PER': per_val, 'PEG': per_val/growth if growth > 0 else 0})

                    elif latest_month in [7, 8, 9]:
                        analysis_type = "[미래 1Q 포함] Forward PEG"
                        base_date = latest_date + pd.DateOffset(months=3)
                        f1_ttm = eps_df['Quarterly_EPS'].iloc[-3:].sum() + curr_q_est
                        per_f1 = current_price / f1_ttm
                        for y in range(5, 0, -1):
                            target_idx = (latest_idx - (y * 4)) + 1
                            if target_idx >= 3:
                                past_ttm = get_ttm(target_idx)
                                if past_ttm > 0:
                                    growth = ((f1_ttm / past_ttm) ** (1/y) - 1) * 100
                                    results.append({'분석 기간': f"최근 {y}년(미래1Q포함)", '과거 TTM': past_ttm, '기준 TTM': f1_ttm, '성장률': growth, 'PER': per_f1, 'PEG': per_f1/growth if growth > 0 else 0})

                    elif latest_month in [4, 5, 6]:
                        analysis_type = "[미래 2Q 포함] Forward PEG"
                        base_date = latest_date + pd.DateOffset(months=6)
                        f2_ttm = eps_df['Quarterly_EPS'].iloc[-2:].sum() + curr_q_est + next_q_est
                        per_f2 = current_price / f2_ttm
                        for y in range(5, 0, -1):
                            target_idx = (latest_idx - (y * 4)) + 2
                            if target_idx >= 3:
                                past_ttm = get_ttm(target_idx)
                                if past_ttm > 0:
                                    growth = ((f2_ttm / past_ttm) ** (1/y) - 1) * 100
                                    results.append({'분석 기간': f"최근 {y}년(미래2Q포함)", '과거 TTM': past_ttm, '기준 TTM': f2_ttm, '성장률': growth, 'PER': per_f2, 'PEG': per_f2/growth if growth > 0 else 0})
                    else:
                        st.info("ℹ️ 연초(1-3월) 데이터이므로 야후 파이낸스 연간 추정치로 분석합니다.")
                        analysis_type = "[추정치 기반] 5년 장기 PEG"
                        curr_per = current_price / curr_year_est
                        target_idx_5y = latest_idx - (5 * 4)
                        if target_idx_5y >= 3:
                            past_ttm_5y = get_ttm(target_idx_5y)
                            if past_ttm_5y > 0:
                                growth_5y = ((curr_year_est / past_ttm_5y) ** (1/5) - 1) * 100
                                results.append({'분석 기간': '5년 장기 추세', '과거 TTM': past_ttm_5y, '기준 TTM': curr_year_est, '성장률': growth_5y, 'PER': curr_per, 'PEG': curr_per/growth_5y if growth_5y > 0 else 0})

                    if results:
                        st.subheader(f"📌 {analysis_type}")
                        st.caption(f"기준일: {base_date.strftime('%Y-%m-%d')} | 현재가: ${current_price:.2f}")
                        
                        df_res = pd.DataFrame(results)
                        df_res.columns = ['분석 기간', '과거 TTM EPS', '기준 TTM EPS', '연평균성장률(%)', 'PER', 'PEG']
                        
                        st.dataframe(df_res.style.format({
                            '과거 TTM EPS': '{:.2f}', 
                            '기준 TTM EPS': '{:.2f}',
                            '연평균성장률(%)': '{:.2f}%', 
                            'PER': '{:.2f}', 
                            'PEG': '{:.2f}'
                        }).highlight_between(left=0.1, right=1.0, subset=['PEG'], color='#D4EDDA'), 
                        width=550, hide_index=True)
                        
                        st.success("✅ 분석 완료: PEG가 1.0 미만인 구간은 초록색으로 표시됩니다.")
                    else:
                        st.warning("⚠️ 분석에 충분한 과거 실적 데이터가 없습니다.")
        except Exception as e:
            st.error(f"❌ 분석 중 오류 발생: {e}")
            st.info("팁: 티커가 올바른지, 혹은 사이트 구조가 변경되었는지 확인하세요.")

# --- 메뉴 5: 기업 가치 비교 ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            ticker_input = st.text_input("🏢 티커 입력", "AAPL, MSFT, GOOGL")
        with col2:
            start_year = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3:
            predict_mode = st.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True, index=0)
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
                    if f_count > 0:
                        ax.plot(range(h_end-1, len(series)), series.values[h_end-1:], linestyle='--', color=colors[i], linewidth=2.0, alpha=0.8)
                apply_strong_style(ax, f"Relative PER Change since {start_year}", "Change (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_xticks(range(len(indexed_per))); ax.set_xticklabels(x_labels, rotation=45)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
                st.pyplot(fig)
        else: # EPS
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
                    valid_data = plot_df[plot_df[t].notna()]
                    if valid_data.empty: continue
                    norm_vals = (plot_df[t] / valid_data[t].iloc[0] - 1) * 100
                    color = plt.cm.Set1(i % 9)
                    act_mask = plot_df['type'] == 'Actual'
                    last_act = np.where(act_mask)[0][-1] if any(act_mask) else 0
                    ax.plot(range(last_act + 1), norm_vals.iloc[:last_act + 1], marker='o', label=f"{t} ({norm_vals.dropna().values[-1]:+.1f}%)", color=color, linewidth=2.5)
                    if predict_mode != "None":
                        ax.plot(range(last_act, len(filtered_idx)), norm_vals.iloc[last_act:], linestyle='--', color=color, linewidth=2.0)
                apply_strong_style(ax, f"Normalized EPS Growth since {start_year}-Q1", "Growth (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax.set_xticks(range(len(filtered_idx))); ax.set_xticklabels(filtered_idx, rotation=45)
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
                st.pyplot(fig)

# --- 메뉴 6: ETF 섹터 수익률 분석 ---
else:
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            sector_list = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
            selected_etfs = st.multiselect("🌐 ETF 선택", sector_list, default=["SPY", "QQQ", "XLK"])
        with col2:
            start_year_etf = st.number_input("📅 기준 연도", 2010, 2025, 2020)
        with col3:
            start_q_etf = st.selectbox("🔢 기준 분기", [1, 2, 3, 4], index=0)
        run_etf_btn = st.button("ETF 수익률 분석 시작", type="primary", use_container_width=True)

    if run_etf_btn and selected_etfs:
        df_etf = fetch_etf_data(selected_etfs)
        start_date = f"{start_year_etf}-{str((start_q_etf-1)*3 + 1).zfill(2)}"
        if any(df_etf.index >= start_date):
            valid_start = df_etf.index[df_etf.index >= start_date][0]
            norm_etf = (df_etf.loc[valid_start:] / df_etf.loc[valid_start:].iloc[0] - 1) * 100
            last_vals = norm_etf.iloc[-1].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
            vivid_colors = plt.cm.get_cmap('tab10', len(selected_etfs))
            for i, ticker in enumerate(last_vals.index):
                lw = 4.0 if ticker in ["SPY", "QQQ"] else 2.5
                ax.plot(norm_etf.index, norm_etf[ticker], label=f"{ticker} ({last_vals[ticker]:+.1f}%)", color=vivid_colors(i), linewidth=lw)
            apply_strong_style(ax, f"ETF Performance since {valid_start}", "Return (%)")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ticks = [d for d in norm_etf.index if d.endswith(('-01', '-04', '-07', '-10'))]
            ax.set_xticks(ticks); ax.set_xticklabels(ticks, rotation=45)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
            st.pyplot(fig)
