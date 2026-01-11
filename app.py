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

# 공통: 소수점 2자리 강제 포맷팅 함수 (문자열 반환)
def fmt(val):
    try:
        return "{:.2f}".format(float(val))
    except:
        return str(val)

# 데이터프레임 전체에 fmt 적용 함수
def format_df(df):
    return df.map(lambda x: fmt(x) if isinstance(x, (int, float)) else x)

# -----------------------------------------------------------
# [Module 1] 개별 종목 밸류에이션 (File #6 & #10 통합)
# -----------------------------------------------------------
def run_single_valuation():
    st.header("💎 개별 종목 밸류에이션")
    
    # 1. UI 입력
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ticker = st.text_input("티커 입력 (예: TSLA)", "TSLA").upper().strip()
    with col2:
        base_year_input = st.selectbox("File 6: 차트 기준 연도", range(2017, 2026), index=0)
    with col3:
        include_est = st.radio("미래 예측치(Estimates) 포함", ["None", "Current Q", "Next Q"], horizontal=True)

    if ticker:
        st.info(f"[{ticker}] 데이터를 분석 중입니다. ChoiceStock 및 Yahoo Finance 데이터를 동기화합니다...")
        
        try:
            # --- [A] ChoiceStock에서 EPS 수집 ---
            url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            
            try:
                dfs = pd.read_html(io.StringIO(response.text))
            except ValueError:
                st.error("해당 종목의 재무 데이터를 찾을 수 없습니다."); return

            eps_df = pd.DataFrame()
            for df in dfs:
                if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                    target = df.set_index(df.columns[0]).transpose()
                    eps_df = target.iloc[:, [0]].copy()
                    eps_df.columns = ['EPS']
                    break
            
            if eps_df.empty:
                st.error("EPS 데이터를 추출할 수 없습니다."); return

            # 날짜 처리
            eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
            eps_df = eps_df.dropna().sort_index()

            # 회계 분기 매칭을 위한 날짜 보정
            def adjust_date(dt):
                return (dt.replace(day=1) - timedelta(days=1)).strftime('%Y-%m') if dt.day <= 5 else dt.strftime('%Y-%m')
            
            # File 6용 월별 날짜 (문자열 인덱스)
            eps_df_monthly = eps_df.copy()
            eps_df_monthly.index = [adjust_date(d) for d in eps_df_monthly.index]
            eps_df_monthly['EPS'] = pd.to_numeric(eps_df_monthly['EPS'].astype(str).str.replace(',', ''), errors='coerce')

            # File 10용 날짜 (Timestamp 인덱스, 2017 이후)
            eps_df_raw = eps_df.copy()
            eps_df_raw.columns = ['EPS']
            eps_df_raw['EPS'] = pd.to_numeric(eps_df_raw['EPS'].astype(str).str.replace(',', ''), errors='coerce')
            eps_df_raw = eps_df_raw[eps_df_raw.index >= "2017-01-01"]

            # --- [B] Yahoo Finance 주가 수집 ---
            stock = yf.Ticker(ticker)
            
            # File 6용 월봉 데이터
            price_month = stock.history(start="2017-01-01", interval="1mo", auto_adjust=False)
            price_month.index = price_month.index.tz_localize(None).strftime('%Y-%m')
            price_month = price_month[['Close']].copy()
            price_month = price_month[~price_month.index.duplicated(keep='last')]
            
            # File 10용 일봉 데이터
            price_daily = stock.history(start="2017-01-01", interval="1d")
            if price_daily.index.tz is not None:
                price_daily.index = price_daily.index.tz_localize(None)
            price_daily_series = price_daily['Close']
            
            # 현재가
            if not price_daily.empty:
                current_price = price_daily['Close'].iloc[-1]
            else:
                st.error("주가 데이터를 가져올 수 없습니다."); return

            # --- [C] 탭 구성 ---
            tab1, tab2 = st.tabs(["📉 연도별 시뮬레이션 (File 6)", "📊 4분기 실적 기반 분석 (File 10)"])

            # -------------------------------------------------------
            # Tab 1: File 6 Logic (연도별 적정주가)
            # -------------------------------------------------------
            with tab1:
                # 데이터 병합 (EPS + Price)
                combined = pd.merge(eps_df_monthly, price_month, left_index=True, right_index=True, how='inner')
                combined = combined.sort_index(ascending=True)
                
                # 미래 예측치 추가
                if include_est != "None":
                    est = stock.earnings_estimate
                    if est is not None and not est.empty:
                        last_date_obj = pd.to_datetime(combined.index[-1])
                        try:
                            curr_val = est['avg'].iloc[0]
                            date_curr = (last_date_obj + pd.DateOffset(months=3)).strftime('%Y-%m')
                            combined.loc[f"{date_curr} (Est.)"] = [curr_val, current_price]
                            if include_est == "Next Q" and len(est) > 1:
                                next_val = est['avg'].iloc[1]
                                date_next = (last_date_obj + pd.DateOffset(months=6)).strftime('%Y-%m')
                                combined.loc[f"{date_next} (Est.)"] = [next_val, current_price]
                        except: pass

                # 시뮬레이션 루프
                summary_data_6 = []
                selected_plot_data = None
                selected_scale_factor = 0

                for base_year in range(2017, 2026):
                    df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                    if len(df_plot) < 2 or df_plot.iloc[0]['EPS'] <= 0: continue

                    scale_factor = df_plot.iloc[0]['Close'] / df_plot.iloc[0]['EPS']
                    df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                    
                    final_fair = df_plot.iloc[-1]['Fair_Value']
                    gap = ((current_price - final_fair) / final_fair) * 100
                    status = "고평가" if gap > 0 else "저평가"

                    summary_data_6.append({
                        "기준 연도": base_year,
                        "적용 PER": scale_factor,
                        "적정 주가": final_fair,
                        "현재 주가": current_price,
                        "괴리율 (%)": gap,
                        "판단": status
                    })

                    if base_year == base_year_input:
                        selected_plot_data = df_plot
                        selected_scale_factor = scale_factor

                # 그래프 및 표 출력
                if selected_plot_data is not None:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(selected_plot_data.index, selected_plot_data['Close'], label='Market Price', color='#1f77b4', marker='o')
                    ax.plot(selected_plot_data.index, selected_plot_data['Fair_Value'], label=f'Fair Value (Base {base_year_input})', color='#d62728', linestyle='--', marker='s')
                    
                    est_idx = [i for i, idx in enumerate(selected_plot_data.index) if "(Est.)" in idx]
                    if est_idx:
                        for i in est_idx:
                            ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.2)
                    
                    ax.set_title(f"Price vs Fair Value (Base: {base_year_input})")
                    plt.xticks(rotation=45); ax.legend(); st.pyplot(fig)
                
                if summary_data_6:
                    st.write(f"### 📋 연도별 시뮬레이션 요약")
                    st.table(format_df(pd.DataFrame(summary_data_6)))

            # -------------------------------------------------------
            # Tab 2: File 10 Logic (4분기 실적 기반 적정주가)
            # -------------------------------------------------------
            with tab2:
                # 1. Target EPS 계산 (최근 3개 확정 + 현재 분기 예측)
                est = stock.earnings_estimate
                curr_q_est = est['avg'].iloc[0] if (est is not None and not est.empty) else 0
                
                if len(eps_df_raw) >= 3:
                    recent_3_actuals = eps_df_raw['EPS'].iloc[-3:].sum()
                    final_target_eps = recent_3_actuals + curr_q_est
                else:
                    st.warning("데이터 부족으로 Target EPS를 계산할 수 없습니다."); return

                # 2. 4분기 단위 루프 분석
                processed_data_10 = []
                # File 10 로직: 0부터 4씩 건너뛰며 그룹화
                for i in range(0, len(eps_df_raw) - 3, 4):
                    group = eps_df_raw.iloc[i:i+4]
                    eps_sum = group['EPS'].sum()
                    s_date, e_date = group.index[0], group.index[-1]
                    
                    # 해당 기간 평균 주가
                    avg_price = price_daily_series[s_date:e_date].mean()
                    if pd.isna(avg_price): continue
                    
                    per = avg_price / eps_sum if eps_sum > 0 else 0
                    
                    # File 10 핵심: 과거의 PER을 현재 Target EPS에 적용
                    fair_value_now = final_target_eps * per
                    gap_pct = ((current_price / fair_value_now) - 1) * 100 if fair_value_now else 0
                    status = "저평가" if current_price < fair_value_now else "고평가"
                    
                    period_str = f"{s_date.year}.{s_date.month} ~ {e_date.year}.{e_date.month}"
                    
                    processed_data_10.append({
                        "기간": period_str,
                        "EPS 합계": eps_sum,
                        "평균 주가": avg_price,
                        "평균 PER": per,
                        "적정 주가 (현재 기준)": fair_value_now,
                        "괴리율 (%)": gap_pct,
                        "판단": status
                    })
                
                if processed_data_10:
                    df_10 = pd.DataFrame(processed_data_10)
                    
                    # 요약 통계
                    valid_pers = [d['평균 PER'] for d in processed_data_10 if d['평균 PER'] > 0]
                    avg_past_per = np.mean(valid_pers) if valid_pers else 0
                    cur_fair_final = final_target_eps * avg_past_per
                    cur_gap_final = ((current_price / cur_fair_final) - 1) * 100 if cur_fair_final else 0
                    
                    st.markdown(f"""
                    ### 🎯 분석 요약
                    * **분석 기준 EPS (Target EPS):** ${fmt(final_target_eps)} (최근 3분기 실적 + 현재 분기 예측)
                    * **현재 주가:** ${fmt(current_price)}
                    * **과거 평균 PER 기준 적정가:** ${fmt(cur_fair_final)} (평균 PER: {fmt(avg_past_per)}배)
                    * **상태:** **{fmt(abs(cur_gap_final))}% {"저평가" if current_price < cur_fair_final else "고평가"}**
                    """)
                    
                    st.write("### 📋 기간별 PER 적용 시뮬레이션")
                    st.table(format_df(df_10))
                else:
                    st.warning("분석할 과거 데이터 구간이 충분하지 않습니다.")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# -----------------------------------------------------------
# [Module 2] 종목 비교 분석 (Files #9, #13)
# -----------------------------------------------------------
def normalize_to_standard_quarter(dt):
    month, year = dt.month, dt.year
    if month in [1, 2, 3]:   new_month = 3
    elif month in [4, 5, 6]: new_month = 6
    elif month in [7, 8, 9]: new_month = 9
    else:                    new_month = 12
    return pd.Timestamp(year=year, month=new_month, day=1) + pd.offsets.MonthEnd(0)

@st.cache_data(ttl=3600)
def fetch_comp_data(ticker, show_q1, show_q2):
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('PER').any()), None)
        if target_df is None: return None, None
        
        per_raw = pd.to_numeric(target_df[target_df.index.str.contains('PER')].transpose().iloc[:, 0], errors='coerce')
        eps_raw = pd.to_numeric(target_df[target_df.index.str.contains('EPS')].transpose().iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
        combined = pd.DataFrame({'PER': per_raw, 'EPS': eps_raw}).dropna()
        combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
        combined = combined.sort_index()
        
        if show_q1:
            stock = yf.Ticker(ticker)
            current_price = stock.history(period="1d")['Close'].iloc[-1]
            est = stock.earnings_estimate
            if est is not None and not est.empty:
                historical_eps = combined['EPS'].tolist()
                q1_dt = combined.index[-1] + pd.DateOffset(months=3)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1 if ttm_eps_q1 != 0 else 0
                combined.loc[q1_dt, 'EPS'] = ttm_eps_q1
                
                if show_q2:
                    q2_dt = q1_dt + pd.DateOffset(months=3)
                    ttm_eps_q2 = sum(historical_eps[-2:]) + est.loc['0q', 'avg'] + est.loc['+1q', 'avg']
                    combined.loc[q2_dt, 'PER'] = current_price / ttm_eps_q2 if ttm_eps_q2 != 0 else 0
                    combined.loc[q2_dt, 'EPS'] = ttm_eps_q2

        combined.index = combined.index.map(normalize_to_standard_quarter)
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined['PER'], combined['EPS']
    except: return None, None

def run_comparison():
    st.header("⚖️ 종목 간 지표 비교 (Sync & Forecast)")
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_input = st.text_input("비교 티커 (쉼표 구분)", "AAPL, MSFT, NVDA")
        t_list = [x.strip().upper() for x in tickers_input.split(',') if x.strip()]
    with col2:
        include_est_comp = st.radio("예측치 포함 (비교)", ["None", "Current Q", "Next Q"], horizontal=True)

    comp_mode = st.selectbox("비교 지표 선택", ["상대 PER 추세", "EPS 성장률 비교"])
    start_year = st.number_input("분석 시작 연도", 2010, 2025, 2017)

    if st.button("비교 차트 생성"):
        q1, q2 = (include_est_comp in ["Current Q", "Next Q"]), (include_est_comp == "Next Q")
        master_df = pd.DataFrame()
        
        for t in t_list:
            per_s, eps_s = fetch_comp_data(t, q1, q2)
            if per_s is not None:
                master_df[t] = per_s if comp_mode == "상대 PER 추세" else eps_s

        if not master_df.empty:
            master_df = master_df[master_df.index >= f"{start_year}-01-01"].sort_index()
            indexed_df = (master_df / master_df.iloc[0]) * 100
            
            fig, ax = plt.subplots(figsize=(15, 8))
            x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_df.index]
            
            for ticker in indexed_df.columns:
                series = indexed_df[ticker].dropna()
                valid_indices = [indexed_df.index.get_loc(dt) for dt in series.index]
                forecast_count = (1 if q1 else 0) + (1 if q2 else 0)
                last_val = series.iloc[-1]
                label_txt = f"{ticker} ({last_val:.2f})"

                if forecast_count > 0 and len(valid_indices) > forecast_count:
                    ax.plot(valid_indices[:-forecast_count], series.values[:-forecast_count], marker='o', label=label_txt)
                    ax.plot(valid_indices[-forecast_count-1:], series.values[-forecast_count-1:], ls='--', marker='x', alpha=0.7)
                else:
                    ax.plot(valid_indices, series.values, marker='o', label=label_txt)
            
            ax.set_xticks(range(len(indexed_df))); ax.set_xticklabels(x_labels, rotation=45)
            ax.axhline(100, color='black', alpha=0.5, ls='--')
            ax.set_title(f"Compare: {comp_mode} (Base 100)")
            ax.legend(); st.pyplot(fig)
        else: st.error("유효한 데이터가 없습니다.")

# -----------------------------------------------------------
# [Module 3] 섹터 수익률 분석 (File #12)
# -----------------------------------------------------------
def run_sector_perf():
    st.header("📊 섹터 및 지수 수익률 분석 (분기 기준)")
    all_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
    selected = st.multiselect("분석할 ETF 선택", all_tickers, default=["SPY", "QQQ", "XLK"])
    
    col1, col2 = st.columns(2)
    with col1: sel_year = st.selectbox("시작 연도", range(2017, datetime.now().year + 1))
    with col2: sel_quarter = st.selectbox("시작 분기", [1, 2, 3, 4])
    
    q_map = {1: "-01-01", 2: "-04-01", 3: "-07-01", 4: "-10-01"}
    start_date_str = f"{sel_year}{q_map[sel_quarter]}"

    if st.button("수익률 차트 생성"):
        combined_price = pd.DataFrame()
        for t in selected:
            df = yf.Ticker(t).history(start="2017-01-01", interval="1mo", auto_adjust=True)
            if not df.empty:
                df.index = df.index.strftime('%Y-%m-%d')
                combined_price[t] = df['Close']
        
        if not combined_price.empty:
            available_dates = combined_price.index[combined_price.index >= start_date_str]
            if len(available_dates) == 0: st.error("데이터 없음"); return
            
            base_date = available_dates[0]
            norm_df = (combined_price.loc[base_date:] / combined_price.loc[base_date]) * 100
            
            fig, ax = plt.subplots(figsize=(15, 8))
            last_val_idx = norm_df.iloc[-1].sort_values(ascending=False)
            for ticker in last_val_idx.index:
                lw = 4 if ticker in ["SPY", "QQQ"] else 2
                zo = 5 if ticker in ["SPY", "QQQ"] else 2
                ax.plot(norm_df.index, norm_df[ticker], label=f"{ticker} ({last_val_idx[ticker]:.2f})", linewidth=lw, zorder=zo)
            
            q_ticks = [d for d in norm_df.index if d.endswith(('-01-01', '-04-01', '-07-01', '-10-01'))]
            ax.set_xticks(q_ticks if q_ticks else norm_df.index[::3])
            plt.xticks(rotation=45); ax.axhline(100, color='black', ls='--')
            ax.set_title(f"ETF Performance (Base: {base_date} = 100)"); ax.legend(); st.pyplot(fig)
            
            st.write(f"### 🏆 {base_date} 이후 누적 수익률 (%)")
            performance_pct = (last_val_idx - 100).to_frame(name="수익률 (%)")
            st.table(format_df(performance_pct))

# -----------------------------------------------------------
# [Main] 메인 메뉴 컨트롤러
# -----------------------------------------------------------
def main():
    st.sidebar.title("🇺🇸 주식 분석 터미널")
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("메뉴 선택", ["홈", "개별 종목 밸류에이션", "종목 비교 분석", "섹터/지수 수익률"])
    
    if menu == "홈":
        st.title("US Stock Analysis System")
        st.markdown("""
        ### 환영합니다!
        이 시스템은 **ChoiceStock**의 재무 데이터와 **Yahoo Finance**의 시장 데이터를 결합하여 분석합니다.
        
        #### 📌 주요 기능
        1. **개별 종목 밸류에이션**:
            * **Tab 1**: 연도별 적정주가 시뮬레이션 (File #6)
            * **Tab 2**: 4분기 실적 기반 분석 (File #10) 

[Image of financial statement analysis]

        2. **종목 비교 분석**: PER 및 EPS 성장 추세 비교 (File #9, #13)
        3. **섹터 수익률**: 분기별 ETF 누적 수익률 (File #12)
        """)
    elif menu == "개별 종목 밸류에이션": run_single_valuation()
    elif menu == "종목 비교 분석": run_comparison()
    elif menu == "섹터/지수 수익률": run_sector_perf()

if __name__ == "__main__":
    main()
