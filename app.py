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
# [Module 2] 종목 비교 분석 (Quarter Sync + 성장률 % 반영)
# -----------------------------------------------------------
def get_future_estimates_yf(ticker):
    try:
        stock = yf.Ticker(ticker)
        est = stock.earnings_estimate
        if est is not None and not est.empty:
            curr_est = est['avg'].iloc[0]
            next_est = est['avg'].iloc[1] if len(est) > 1 else None
            return {'current': curr_est, 'next': next_est}
    except: pass
    return None

def _get_ticker_data_integrated(ticker, include_mode):
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS').any()), None)
        if target_df is None: return pd.DataFrame()
        
        target_df = target_df.set_index(target_df.columns[0]).transpose()
        eps_df = target_df.iloc[:, [0]].copy()
        eps_df.columns = [ticker]
        eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
        eps_df = eps_df.dropna()
        
        def to_quarter_label(dt):
            actual_dt = (dt.replace(day=1) - timedelta(days=1)) if dt.day <= 5 else dt
            return f"{actual_dt.year}-Q{(actual_dt.month-1)//3 + 1}"

        eps_df.index = [to_quarter_label(d) for d in eps_df.index]
        eps_df[ticker] = pd.to_numeric(eps_df[ticker].astype(str).str.replace(',', ''), errors='coerce')
        eps_df = eps_df.groupby(level=0).last()
        eps_df['type'] = 'Actual'

        # 예측치 선택 로직 적용
        if include_mode != "None":
            estimates = get_future_estimates_yf(ticker)
            if estimates:
                last_q = eps_df.index[-1]
                year, q = int(last_q.split('-Q')[0]), int(last_q.split('-Q')[1])
                
                # Current Q 추가
                if estimates['current'] is not None:
                    new_q_val = q + 1
                    new_year = year + (new_q_val - 1) // 4
                    q_label = f"{new_year}-Q{(new_q_val - 1) % 4 + 1}"
                    eps_df.loc[q_label, ticker] = estimates['current']
                    eps_df.loc[q_label, 'type'] = 'Estimate'
                
                # Next Q 추가 (옵션이 Next Q일 때만)
                if include_mode == "Next Q" and estimates['next'] is not None:
                    new_q_val = q + 2
                    new_year = year + (new_q_val - 1) // 4
                    q_label = f"{new_year}-Q{(new_q_val - 1) % 4 + 1}"
                    eps_df.loc[q_label, ticker] = estimates['next']
                    eps_df.loc[q_label, 'type'] = 'Estimate'
                    
        return eps_df
    except: return pd.DataFrame()

def run_comparison():
    st.header("⚖️ 종목 간 EPS 성장률 비교 (Quarter Sync)")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tickers_input = st.text_input("비교 티커 (쉼표 구분)", "SNPS, FDS, GOOGL")
        t_list = [x.strip().upper() for x in tickers_input.replace(',', ' ').split() if x.strip()]
    with col2:
        start_year = st.number_input("비교 시작 연도", 2010, 2025, 2020)
    with col3:
        include_mode = st.radio("예측치 선택", ["None", "Current Q", "Next Q"], horizontal=True)

    if st.button("성장률 차트 생성"):
        all_data = []
        for t in t_list:
            df = _get_ticker_data_integrated(t, include_mode)
            if not df.empty: all_data.append(df)
        
        if not all_data:
            st.error("데이터를 불러오지 못했습니다."); return

        combined_index = sorted(list(set().union(*(d.index for d in all_data))))
        combined_index = [i for i in combined_index if i >= f"{start_year}-Q1"]

        fig, ax = plt.subplots(figsize=(12, 6))
        for df in all_data:
            ticker = [c for c in df.columns if c != 'type'][0]
            base_data = df[df.index >= f"{start_year}-Q1"]
            if base_data.empty: continue
            
            base_val = base_data[ticker].dropna().iloc[0]
            plot_df = df.reindex(combined_index)
            norm_values = plot_df[ticker] / base_val
            
            actual_mask = plot_df['type'] == 'Actual'
            est_mask = plot_df['type'] == 'Estimate'
            
            # 최종 성장률 계산 (%)
            final_val = norm_values.dropna().iloc[-1]
            growth_pct = (final_val - 1) * 100
            
            # 범례 레이블 수정: (Est.) 삭제 및 성장률 % 표시
            label_text = f"{ticker} (Actual) {growth_pct:+.1f}%"
            
            # 실제 데이터 그리기 (실선)
            x_actual = [combined_index.index(i) for i in plot_df[actual_mask].index]
            line = ax.plot(x_actual, norm_values[actual_mask], marker='o', label=label_text, linewidth=2)
            
            # 예측 데이터 연결 그리기 (점선) - 범례에서는 제외됨
            if est_mask.any():
                last_act_idx = plot_df[actual_mask].index[-1]
                est_indices = [last_act_idx] + list(plot_df[est_mask].index)
                x_est = [combined_index.index(i) for i in est_indices]
                ax.plot(x_est, norm_values[est_indices], ls='--', marker='x', color=line[0].get_color(), alpha=0.7)

        ax.set_xticks(range(len(combined_index)))
        ax.set_xticklabels(combined_index, rotation=45)
        ax.set_ylabel(f"Normalized Growth (Base: {start_year}-Q1 = 1.0)")
        ax.set_title(f"EPS Growth Comparison (Base: {start_year})")
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
        st.info("결산월 자동 보정 기능 및 예측치 선택 기능이 적용되었습니다.")
    elif menu == "개별 종목 밸류에이션": run_single_valuation()
    elif menu == "종목 비교 분석": run_comparison()
    elif menu == "섹터 수익률": run_sector_perf()

if __name__ == "__main__":
    main()
