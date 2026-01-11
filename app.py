import streamlit as st
import pandas as pd
import yfinance as yf
import io
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
import numpy as np
import warnings

# -----------------------------------------------------------
# [0] 환경 설정 및 공통 유틸리티
# -----------------------------------------------------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="미국주식 통합 분석 시스템", layout="wide")
plt.style.use('seaborn-v0_8-whitegrid')

def fmt(val):
    try: return "{:.2f}".format(float(val))
    except: return str(val)

def format_df(df):
    return df.map(lambda x: fmt(x) if isinstance(x, (int, float)) else x)

def normalize_to_standard_quarter(dt):
    """서로 다른 분기 마감일을 가장 가까운 표준 분기(3, 6, 9, 12월)로 조정"""
    month, year = dt.month, dt.year
    if month in [1, 2, 3]:   new_month = 3
    elif month in [4, 5, 6]: new_month = 6
    elif month in [7, 8, 9]: new_month = 9
    else:                    new_month = 12
    return pd.Timestamp(year=year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

# -----------------------------------------------------------
# [Module 1] 개별 종목 밸류에이션 (기존 기능 유지)
# -----------------------------------------------------------
def run_single_valuation():
    st.header("💎 개별 종목 밸류에이션")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ticker = st.text_input("티커 입력 (예: TSLA)", "TSLA").upper().strip()
    with col2:
        base_year_input = st.selectbox("기준 연도", range(2017, 2026), index=0)
    with col3:
        include_est = st.radio("예측치 포함", ["None", "Current Q", "Next Q"], horizontal=True)

    if ticker:
        try:
            url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            dfs = pd.read_html(io.StringIO(response.text))

            eps_df_raw = pd.DataFrame()
            for df in dfs:
                if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                    target = df.set_index(df.columns[0]).transpose()
                    eps_df_raw = target.iloc[:, [0]].copy()
                    eps_df_raw.columns = ['EPS']
                    break
            
            if eps_df_raw.empty:
                st.error("데이터를 찾을 수 없습니다."); return

            eps_df_raw.index = pd.to_datetime(eps_df_raw.index, format='%y.%m.%d', errors='coerce')
            eps_df_raw = eps_df_raw.dropna().sort_index()

            stock = yf.Ticker(ticker)
            price_daily = stock.history(start="2017-01-01")['Close']
            current_price = price_daily.iloc[-1]

            tab1, tab2 = st.tabs(["📉 연도별 시뮬레이션", "📊 4분기 실적 기반 분석"])

            with tab1:
                combined = eps_df_raw.copy()
                combined.index = combined.index.strftime('%Y-%m')
                price_m = stock.history(start="2017-01-01", interval="1mo")['Close']
                price_m.index = price_m.index.tz_localize(None).strftime('%Y-%m')
                combined = pd.merge(combined, price_m, left_index=True, right_index=True, how='inner')

                summary_data = []
                for by in range(2017, 2026):
                    df_p = combined[combined.index >= f'{by}-01'].copy()
                    if df_p.empty or df_p.iloc[0]['EPS'] <= 0: continue
                    sf = df_p.iloc[0]['Close'] / df_p.iloc[0]['EPS']
                    df_p['Fair'] = df_p['EPS'] * sf
                    gap = ((current_price - df_p['Fair'].iloc[-1]) / df_p['Fair'].iloc[-1]) * 100
                    summary_data.append({"기준년도": by, "PER": sf, "적정가": df_p['Fair'].iloc[-1], "현재가": current_price, "괴리율": gap})
                    if by == base_year_input:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(df_p.index, df_p['Close'], label='Market')
                        ax.plot(df_p.index, df_p['Fair'], label='Fair', ls='--')
                        plt.xticks(rotation=45); st.pyplot(fig)
                st.table(format_df(pd.DataFrame(summary_data)))

            with tab2:
                est = stock.earnings_estimate
                target_eps = eps_df_raw['EPS'].iloc[-3:].sum() + (est['avg'].iloc[0] if est is not None else 0)
                res10 = []
                for i in range(0, len(eps_df_raw)-3, 4):
                    grp = eps_df_raw.iloc[i:i+4]
                    e_sum = grp['EPS'].sum()
                    avg_p = price_daily[grp.index[0]:grp.index[-1]].mean()
                    per = avg_p / e_sum if e_sum > 0 else 0
                    fair = target_eps * per
                    res10.append({"기간": f"{grp.index[0].year}-{grp.index[-1].year}", "PER": per, "적정가": fair, "판단": "저평가" if current_price < fair else "고평가"})
                st.table(format_df(pd.DataFrame(res10)))
        except Exception as e:
            st.error(f"오류 발생: {e}")

# -----------------------------------------------------------
# [Module 2] 종목 비교 분석 (Quarter Sync PER & EPS 통합)
# -----------------------------------------------------------
def fetch_multicycle_ticker_per(ticker, show_q1, show_q2):
    """표준 분기 동기화 및 예측 PER 계산 함수"""
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
        historical_eps = combined['EPS'].tolist()
        
        # 야후 예측치 기반 Forward PER 계산
        if show_q1:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            est = stock.earnings_estimate
            if est is not None and not est.empty:
                last_dt = combined.index[-1]
                # Current Q
                q1_dt = last_dt + pd.DateOffset(months=3)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1 if ttm_eps_q1 != 0 else 0
                combined.loc[q1_dt, 'type'] = 'Estimate'
                # Next Q
                if show_q2:
                    q2_dt = q1_dt + pd.DateOffset(months=3)
                    ttm_eps_q2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[q2_dt, 'PER'] = current_price / ttm_eps_q2 if ttm_eps_q2 != 0 else 0
                    combined.loc[q2_dt, 'type'] = 'Estimate'

        combined['type'] = combined['type'].fillna('Actual')
        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined
    except: return None

def _get_ticker_eps_synced(ticker, include_mode):
    """EPS 비교용 데이터 수집 (Module 2-EPS용)"""
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS').any()), None)
        if target_df is None: return pd.DataFrame()
        
        eps_df = target_df[target_df.index.str.contains('EPS')].transpose()
        eps_df.columns = [ticker]
        eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
        eps_df = eps_df.dropna().sort_index()
        
        def to_label(dt):
            act = (dt.replace(day=1) - timedelta(days=1)) if dt.day <= 5 else dt
            return f"{act.year}-Q{(act.month-1)//3 + 1}"

        res = pd.DataFrame({ticker: pd.to_numeric(eps_df[ticker].astype(str).str.replace(',', ''), errors='coerce')})
        res.index = [to_label(d) for d in res.index]
        res = res.groupby(level=0).last()
        res['type'] = 'Actual'

        if include_mode != "None":
            stock = yf.Ticker(ticker)
            est = stock.earnings_estimate
            if est is not None and not est.empty:
                last_q = res.index[-1]
                y, q = int(last_q.split('-Q')[0]), int(last_q.split('-Q')[1])
                # Current Q
                new_q = q + 1
                q_label = f"{y + (new_q-1)//4}-Q{(new_q-1)%4+1}"
                res.loc[q_label, [ticker, 'type']] = [est['avg'].iloc[0], 'Estimate']
                # Next Q
                if include_mode == "Next Q" and len(est) > 1:
                    new_q = q + 2
                    q_label = f"{y + (new_q-1)//4}-Q{(new_q-1)%4+1}"
                    res.loc[q_label, [ticker, 'type']] = [est['avg'].iloc[1], 'Estimate']
        return res
    except: return pd.DataFrame()

def run_comparison():
    st.header("⚖️ 종목 간 지표 비교 (Standard Quarter Sync)")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        tickers_input = st.text_input("비교 티커 (쉼표 구분)", "SNPS, FDS, GOOGL")
        t_list = [x.strip().upper() for x in tickers_input.replace(',', ' ').split() if x.strip()]
    with col2:
        start_year = st.number_input("비교 시작 연도", 2010, 2025, 2020)
    with col3:
        include_mode = st.radio("예측치 선택", ["None", "Current Q", "Next Q"], horizontal=True)
    with col4:
        comp_target = st.radio("비교 지표", ["상대 PER 추세", "EPS 성장률"])

    if st.button("비교 차트 생성"):
        all_data = []
        show_q1 = (include_mode != "None")
        show_q2 = (include_mode == "Next Q")

        for t in t_list:
            if comp_target == "상대 PER 추세":
                df = fetch_multicycle_ticker_per(t, show_q1, show_q2)
            else:
                df = _get_ticker_eps_synced(t, include_mode)
            if df is not None and not df.empty: all_data.append(df)
        
        if not all_data:
            st.error("데이터가 없습니다."); return

        # X축 레이블 생성 (표준 분기 기반)
        if comp_target == "상대 PER 추세":
            master_index = sorted(list(set().union(*(d.index for d in all_data))))
            master_index = [i for i in master_index if i >= pd.Timestamp(f"{start_year}-01-01")]
            x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in master_index]
        else:
            master_index = sorted(list(set().union(*(d.index for d in all_data))))
            master_index = [i for i in master_index if i >= f"{start_year}-Q1"]
            x_labels = master_index

        fig, ax = plt.subplots(figsize=(12, 6))
        for df in all_data:
            ticker = [c for c in df.columns if c not in ['type', 'EPS']][0]
            
            # 기준 시점 데이터 필터링 및 정규화
            base_df = df[df.index >= master_index[0]]
            if base_df.empty: continue
            
            base_val = base_df[ticker].dropna().iloc[0]
            plot_df = df.reindex(master_index)
            norm_values = (plot_df[ticker] / base_val) * 100
            
            actual_mask = plot_df['type'] == 'Actual'
            est_mask = plot_df['type'] == 'Estimate'
            
            final_val = norm_values.dropna().iloc[-1]
            growth_pct = final_val - 100
            label_text = f"{ticker} (Actual) {growth_pct:+.1f}%"
            
            # 실제 데이터 (실선)
            x_idx = range(len(master_index))
            line = ax.plot([x_idx[i] for i, m in enumerate(actual_mask) if m], 
                           norm_values[actual_mask], marker='o', label=label_text, linewidth=2)
            
            # 예측 데이터 (점선)
            if est_mask.any():
                last_act_pos = [i for i, m in enumerate(actual_mask) if m][-1]
                est_positions = [last_act_pos] + [i for i, m in enumerate(est_mask) if m]
                ax.plot(est_positions, norm_values.iloc[est_positions], ls='--', marker='x', color=line[0].get_color(), alpha=0.7)

        ax.axhline(100, color='black', ls='--', alpha=0.5)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_title(f"{comp_target} Comparison (Base 100 at {start_year})")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

# -----------------------------------------------------------
# [Module 3] 섹터 수익률 분석 (기존 기능 유지)
# -----------------------------------------------------------
def run_sector_perf():
    st.header("📊 섹터 수익률 분석")
    selected = st.multiselect("ETF 선택", ["SPY", "QQQ", "XLK", "XLY", "XLF"], default=["SPY", "QQQ", "XLK"])
    start_date = st.date_input("시작 날짜", datetime(2023, 1, 1))

    if st.button("수익률 확인"):
        prices = pd.DataFrame()
        for t in selected:
            try:
                data = yf.Ticker(t).history(start=start_date)['Close']
                if not data.empty: prices[t] = data
            except: pass
        if not prices.empty:
            norm_prices = (prices / prices.iloc[0]) * 100
            fig, ax = plt.subplots(figsize=(10, 5))
            for c in norm_prices.columns:
                ax.plot(norm_prices.index, norm_prices[c], label=c)
            ax.axhline(100, color='black', ls='--')
            ax.legend(); st.pyplot(fig)

# -----------------------------------------------------------
# [Main] 메인 메뉴 컨트롤러
# -----------------------------------------------------------
def main():
    st.sidebar.title("🇺🇸 주식 분석 터미널")
    menu = st.sidebar.radio("메뉴", ["홈", "개별 종목 밸류에이션", "종목 비교 분석", "섹터 수익률"])
    
    if menu == "홈":
        st.title("통합 분석 시스템")
        st.info("결산월 동기화 기능이 PER 및 EPS 비교에 모두 적용되었습니다.")
    elif menu == "개별 종목 밸류에이션": run_single_valuation()
    elif menu == "종목 비교 분석": run_comparison()
    elif menu == "섹터 수익률": run_sector_perf()

if __name__ == "__main__":
    main()
