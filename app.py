import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import io
import numpy as np
from datetime import datetime, timedelta
import warnings

# 경고 무시 및 기본 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="US Stock Valuation Dashboard", layout="wide")
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

# ==========================================
# [공통 함수] 데이터 수집 및 유틸리티 (캐싱 적용)
# ==========================================

@st.cache_data(ttl=3600)  # 1시간 캐시
def get_choicestock_data(ticker, data_type='EPS'):
    """
    ChoiceStock에서 과거 실적(EPS 또는 PER) 데이터를 크롤링합니다.
    data_type: 'EPS' or 'PER'
    """
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(response.text))
        
        target_df = None
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains(data_type).any():
                target_df = df.set_index(df.columns[0])
                break
        
        if target_df is None:
            return pd.DataFrame()

        # 데이터 정제
        raw_data = target_df[target_df.index.str.contains(data_type, na=False)].transpose()
        raw_data.index = pd.to_datetime(raw_data.index, format='%y.%m.%d', errors='coerce')
        raw_data = raw_data.dropna().sort_index()
        
        # 숫자 변환 (콤마 제거)
        col_name = 'Value'
        raw_data.columns = [col_name]
        raw_data[col_name] = pd.to_numeric(raw_data[col_name].astype(str).str.replace(',', ''), errors='coerce')
        
        # 2017년 이후 데이터만 필터링
        raw_data = raw_data[raw_data.index >= "2017-01-01"]
        
        return raw_data

    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_yahoo_price(ticker, start_date="2017-01-01"):
    """Yahoo Finance 주가 데이터"""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, interval="1d")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df['Close']

@st.cache_data(ttl=3600)
def get_yahoo_estimates(ticker):
    """Yahoo Finance 예측치 (Current Q, Next Q, Current Year)"""
    try:
        stock = yf.Ticker(ticker)
        est = stock.earnings_estimate
        if est is not None and not est.empty:
            return {
                'curr_q': est.loc['0q', 'avg'] if '0q' in est.index else None,
                'next_q': est.loc['+1q', 'avg'] if '+1q' in est.index else None,
                'curr_y': est.loc['0y', 'avg'] if '0y' in est.index else None
            }
    except:
        pass
    return {}

# ==========================================
# [모듈 1] 섹터 및 벤치마크 분석
# ==========================================
def render_sector_analysis():
    st.header("📊 Sector & Benchmark Performance")
    st.markdown("ETF 및 벤치마크 지수의 성과를 비교합니다. (Base = 100)")

    col1, col2 = st.columns([1, 3])
    
    with col1:
        default_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
        selected_tickers = st.multiselect("종목 선택", default_tickers, default=["SPY", "QQQ", "XLK"])
        
        start_year = st.selectbox("시작 연도", range(2017, 2026), index=0) # 2017 default
        start_quarter = st.selectbox("시작 분기", [1, 2, 3, 4], index=0)
        
        run_btn = st.button("분석 실행", key="sector_btn")

    if run_btn and selected_tickers:
        with st.spinner("데이터 수집 중..."):
            combined_df = pd.DataFrame()
            
            # 시작일 계산
            q_map = {1: "-01", 2: "-04", 3: "-07", 4: "-10"}
            start_date_str = f"{start_year}{q_map[start_quarter]}"

            for ticker in selected_tickers:
                stock = yf.Ticker(ticker)
                # 월봉 데이터
                df = stock.history(start="2017-01-01", interval="1mo", auto_adjust=True)
                if df.empty: continue
                
                temp = df[['Close']].copy()
                temp.index = temp.index.strftime('%Y-%m')
                temp = temp[~temp.index.duplicated(keep='first')]
                temp.columns = [ticker]
                
                if combined_df.empty:
                    combined_df = temp
                else:
                    combined_df = combined_df.join(temp, how='outer')

            # 시작 시점 필터링
            if start_date_str not in combined_df.index:
                # 해당 날짜가 없으면 그 이후 가장 빠른 날짜 선택
                valid_dates = combined_df.index[combined_df.index >= start_date_str]
                if len(valid_dates) > 0:
                    start_date_str = valid_dates[0]
                else:
                    st.error("선택한 기간의 데이터가 없습니다.")
                    return

            # 정규화 (Base=100)
            base_row = combined_df.loc[start_date_str]
            normalized_df = ((combined_df.loc[start_date_str:] / base_row) * 100).round(2)

            # 시각화
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 수익률 순으로 정렬하여 범례 표시
            last_val = normalized_df.iloc[-1].sort_values(ascending=False)
            
            for col in last_val.index:
                linewidth = 3 if col in ["SPY", "QQQ"] else 1.5
                alpha = 1.0 if col in ["SPY", "QQQ"] else 0.7
                ax.plot(normalized_df.index, normalized_df[col], 
                        label=f"{col} ({last_val[col]:.1f})", linewidth=linewidth, alpha=alpha)

            ax.axhline(100, color='black', linestyle='--', linewidth=1)
            
            # X축 레이블 간소화
            ticks = [d for d in normalized_df.index if d.endswith(('-01', '-07'))] # 6개월 단위
            ax.set_xticks(ticks)
            plt.xticks(rotation=45)
            
            ax.set_title(f"Performance Comparison (Base: {start_date_str} = 100)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Returns")
            ax.grid(True, linestyle=':', alpha=0.6)
            
            st.pyplot(fig)
            st.dataframe(last_val.to_frame(name="Final Score"))

# ==========================================
# [모듈 2] 개별 종목 정밀 밸류에이션
# ==========================================
def render_valuation_analysis():
    st.header("💎 Single Stock Valuation Master")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        ticker = st.text_input("티커 입력 (예: AAPL)", value="AAPL").upper().strip()
    with col2:
        est_option = st.radio("예측치 포함 범위", ["포함 안함", "현재 분기(Current)", "다음 분기(Next)까지"])
    
    if not ticker:
        return

    # 데이터 로드
    eps_data = get_choicestock_data(ticker, 'EPS')
    price_data = get_yahoo_price(ticker)
    estimates = get_yahoo_estimates(ticker)
    
    if eps_data.empty:
        st.error("EPS 데이터를 불러올 수 없습니다. (ChoiceStock 크롤링 실패 또는 데이터 부족)")
        return
    
    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["Base Year Valuation", "PER Trend (Mean/Median)", "Annual Summary Table", "PEG Analysis"])

    # --- Tab 1: Base Year Valuation (File 2 Logic) ---
    with tab1:
        st.subheader("연도별 적정주가 시뮬레이션")
        st.caption("2017년부터 각 연도를 기준점으로 잡았을 때, 현재 주가가 고평가인지 저평가인지 판단합니다.")
        
        # 데이터 병합
        eps_data.index = eps_data.index.strftime('%Y-%m')
        price_monthly = price_data.resample('M').last()
        price_monthly.index = price_monthly.index.strftime('%Y-%m')
        
        combined = pd.DataFrame({'EPS': eps_data['Value'], 'Close': price_monthly}).dropna()
        
        # 예측치 추가 로직
        if est_option != "포함 안함" and estimates:
            last_date = pd.to_datetime(combined.index[-1])
            curr_p = combined['Close'].iloc[-1]
            
            # Current Q
            if estimates['curr_q']:
                date_curr = (last_date + pd.DateOffset(months=3)).strftime('%Y-%m')
                combined.loc[f"{date_curr} (Est.)"] = [estimates['curr_q'], curr_p]
                
            # Next Q
            if est_option == "다음 분기(Next)까지" and estimates['next_q']:
                date_next = (last_date + pd.DateOffset(months=6)).strftime('%Y-%m')
                combined.loc[f"{date_next} (Est.)"] = [estimates['next_q'], curr_p]

        # 시뮬레이션 루프
        results = []
        final_price = combined['Close'].iloc[-1]
        
        for base_year in range(2017, 2026):
            start_idx = f"{base_year}-01"
            subset = combined[combined.index >= start_idx].copy()
            if len(subset) < 2 or subset.iloc[0]['EPS'] <= 0: continue
            
            base_eps = subset.iloc[0]['EPS']
            base_price = subset.iloc[0]['Close']
            scale_factor = base_price / base_eps
            
            subset['Fair_Value'] = subset['EPS'] * scale_factor
            final_fv = subset['Fair_Value'].iloc[-1]
            gap = ((final_price - final_fv) / final_fv) * 100
            
            results.append({
                "Base Year": base_year,
                "Multiplier": f"{scale_factor:.1f}x",
                "Fair Value": final_fv,
                "Gap (%)": gap,
                "Status": "Overvalued" if gap > 0 else "Undervalued"
            })

        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.format({"Fair Value": "${:.2f}", "Gap (%)": "{:+.2f}%"}), use_container_width=True)
            
            # 가장 최근 유효한 Base Year 그래프 그리기 (예시)
            best_base = res_df.iloc[0]['Base Year'] # 2017년 기준
            subset = combined[combined.index >= f"{best_base}-01"].copy()
            base_eps = subset.iloc[0]['EPS']
            base_price = subset.iloc[0]['Close']
            factor = base_price / base_eps
            subset['Fair_Value'] = subset['EPS'] * factor
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(subset.index, subset['Close'], label='Market Price', color='blue', marker='o')
            ax.plot(subset.index, subset['Fair_Value'], label=f'Fair Value (Base: {best_base}, PER: {factor:.1f}x)', color='red', linestyle='--')
            
            # 예측 구간 표시
            for i, idx in enumerate(subset.index):
                if "(Est.)" in idx:
                    ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.2)
            
            plt.xticks(rotation=45)
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_title(f"Valuation Chart (Base Year: {best_base})")
            st.pyplot(fig)
        else:
            st.warning("분석 가능한 흑자 데이터가 충분하지 않습니다.")

    # --- Tab 2: PER Trend (Mean vs Median) (File 8 Logic) ---
    with tab2:
        st.subheader("PER Band Analysis")
        per_data = get_choicestock_data(ticker, 'PER')
        
        if not per_data.empty:
            # 예측치 반영을 위한 TTM 계산 로직 재구성 필요
            # 여기서는 편의상 크롤링 된 PER 데이터를 메인으로 쓰되, 
            # 예측치가 있다면 마지막 PER를 수정하는 방식으로 근사 구현
            
            plot_df = per_data.copy()
            plot_df.columns = ['PER']
            
            # 통계치
            avg_per = plot_df['PER'].mean()
            med_per = plot_df['PER'].median()
            curr_per = plot_df['PER'].iloc[-1]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(plot_df.index, plot_df['PER'], marker='o', color='#34495e', label='PER Trend')
            ax.axhline(avg_per, color='#e74c3c', linestyle='--', label=f'Mean: {avg_per:.2f}')
            ax.axhline(med_per, color='#8e44ad', linestyle='-.', label=f'Median: {med_per:.2f}')
            
            ax.set_title(f"Historical PER Trend (Current: {curr_per:.2f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("PER 데이터를 찾을 수 없습니다.")

    # --- Tab 3: Annual Summary (File 10 Logic) ---
    with tab3:
        st.subheader("4분기 단위 밸류에이션 요약")
        
        # 최근 3개 분기 확정 EPS + 야후 Current Q Est
        recent_3 = eps_data['Value'].iloc[-3:].sum() if len(eps_data) >= 3 else 0
        curr_q_est = estimates.get('curr_q', 0) if estimates else 0
        target_eps = recent_3 + curr_q_est
        
        st.metric(label="Forward Target EPS (3 Actual + 1 Est)", value=f"${target_eps:.2f}")
        
        # 테이블 생성
        rows = []
        # 4개씩 묶어서 처리
        raw_eps_rev = eps_data.iloc[::-1] # 역순으로 최신부터 처리 가능하지만, 원본 로직 따름 (오래된 순)
        
        # 원본 로직: 발표일 기준 4개씩 묶기
        for i in range(0, len(eps_data) - 3, 4):
            group = eps_data.iloc[i:i+4]
            eps_sum = group['Value'].sum()
            s_date, e_date = group.index[0], group.index[-1]
            
            # 해당 기간 평균 주가
            period_price = price_data[s_date:e_date].mean()
            if pd.isna(period_price): period_price = 0
            
            per = period_price / eps_sum if eps_sum > 0 else 0
            
            # 마지막 행 처리 (예측치 포함)
            if i + 4 >= len(eps_data):
                eps_sum = target_eps
                per = period_price / eps_sum if eps_sum > 0 else 0 # 단순 참고용
            
            rows.append({
                "Period": f"{s_date.year}~{e_date.year}",
                "EPS Sum": eps_sum,
                "Avg Price": period_price,
                "Avg PER": per
            })
            
        summary_df = pd.DataFrame(rows)
        if not summary_df.empty:
            avg_past_per = summary_df[summary_df['Avg PER'] > 0]['Avg PER'].mean()
            fair_value_now = target_eps * avg_past_per
            curr_price = price_data.iloc[-1]
            
            summary_df['Fair Value'] = summary_df['EPS Sum'] * summary_df['Avg PER'] # Self reference for history
            
            st.dataframe(summary_df.style.format("{:.2f}"))
            
            st.info(f"""
            **현재 시점 분석**
            * 현재 주가: **${curr_price:.2f}**
            * 과거 평균 PER 적용 적정가: **${fair_value_now:.2f}** ({avg_past_per:.1f}x 적용)
            * 상태: **{"저평가" if curr_price < fair_value_now else "고평가"}**
            """)

    # --- Tab 4: PEG Analysis (File 11 Logic) ---
    with tab4:
        st.subheader("PEG (Price/Earnings-to-Growth) Analysis")
        # 최근 확정 EPS TTM
        if len(eps_data) >= 4:
            ttm_current = eps_data['Value'].iloc[-4:].sum()
            curr_price = price_data.iloc[-1]
            per_ttm = curr_price / ttm_current
            
            peg_rows = []
            # 5년 전부터 성장률 계산
            for y in range(5, 0, -1):
                idx = len(eps_data) - 1 - (y * 4)
                if idx >= 3:
                    past_ttm = eps_data['Value'].iloc[idx-3:idx+1].sum()
                    if past_ttm > 0:
                        growth = ((ttm_current / past_ttm) ** (1/y) - 1) * 100
                        peg = per_ttm / growth if growth > 0 else np.nan
                        peg_rows.append({
                            "Period": f"{y} Years Ago",
                            "Past TTM": past_ttm,
                            "Current TTM": ttm_current,
                            "CAGR (%)": growth,
                            "PEG": peg
                        })
            
            peg_df = pd.DataFrame(peg_rows)
            st.dataframe(peg_df.style.format({"PEG": "{:.2f}", "CAGR (%)": "{:.2f}%", "Past TTM": "{:.2f}", "Current TTM": "{:.2f}"}))
            
            # Yahoo Est 기반 PEG
            if estimates.get('curr_y'):
                st.markdown("---")
                st.markdown(f"**Yahoo Finance Estimates PEG** (Current Year Est: ${estimates['curr_y']:.2f})")
                fwd_per = curr_price / estimates['curr_y']
                st.write(f"Forward PER: {fwd_per:.2f}")

# ==========================================
# [모듈 3] 비교 분석 (Growth & PER)
# ==========================================
def render_comparison_analysis():
    st.header("⚖️ Stock Comparison Tool")
    
    tickers_input = st.text_input("비교할 티커 입력 (쉼표 구분)", "MSFT, AAPL, GOOGL").upper()
    ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
    
    start_year = st.selectbox("비교 시작 연도", range(2017, 2026), index=3) # 2020 default
    
    tab1, tab2 = st.tabs(["EPS Growth Comparison", "Multi-Cycle PER Comparison"])
    
    # --- Tab 1: EPS Growth (File 13) ---
    with tab1:
        if st.button("EPS 성장률 비교 실행"):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for t in ticker_list:
                df = get_choicestock_data(t, 'EPS')
                if df.empty: continue
                
                # 예측치 통합
                est = get_yahoo_estimates(t)
                combined_s = df['Value'].copy()
                
                if est:
                    last_date = combined_s.index[-1]
                    if est['curr_q']:
                        combined_s.loc[last_date + pd.DateOffset(months=3)] = est['curr_q']
                    if est['next_q']:
                        combined_s.loc[last_date + pd.DateOffset(months=6)] = est['next_q']
                
                # Base Year 필터링 및 정규화
                base_data = combined_s[combined_s.index >= f"{start_year}-01-01"]
                if base_data.empty: continue
                
                # Normalize (Start = 1.0)
                base_val = base_data.iloc[0]
                if base_val <= 0: continue # 적자 기업 제외
                
                norm_data = base_data / base_val
                
                # 실제/예측 구분 시각화
                # 실제 데이터
                actual_mask = norm_data.index <= df.index[-1]
                ax.plot(norm_data[actual_mask].index, norm_data[actual_mask], marker='o', label=f"{t}")
                
                # 예측 데이터 (점선)
                est_mask = norm_data.index >= df.index[-1]
                if len(norm_data[est_mask]) > 1:
                    ax.plot(norm_data[est_mask].index, norm_data[est_mask], linestyle='--', color=ax.lines[-1].get_color())
            
            ax.set_title("Normalized EPS Growth")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # --- Tab 2: PER Compare (File 9) ---
    with tab2:
        if st.button("PER 추세 비교 실행"):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for t in ticker_list:
                per_df = get_choicestock_data(t, 'PER')
                if per_df.empty: continue
                
                # 시작일 이후 데이터
                subset = per_df[per_df.index >= f"{start_year}-01-01"]
                if subset.empty: continue
                
                # 정규화 (Base=100)
                normalized_per = (subset['Value'] / subset['Value'].iloc[0]) * 100
                
                ax.plot(normalized_per.index, normalized_per, label=f"{t} (Last: {subset['Value'].iloc[-1]:.1f})")
                
            ax.axhline(100, color='black', linestyle='--', linewidth=1)
            ax.set_title(f"Relative PER Trend (Base: {start_year} = 100)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ==========================================
# 메인 앱 실행 구조
# ==========================================
def main():
    st.sidebar.title("🇺🇸 US Stock Analytics")
    st.sidebar.info("Developed with ChoiceStock & Yahoo Finance Data")
    
    menu = st.sidebar.radio("Menu", ["Home", "Sector/Market Analysis", "Single Stock Valuation", "Comparison Tool"])
    
    if menu == "Home":
        st.title("Welcome to Investment Dashboard")
        st.markdown("""
        ### 사용 가이드
        이 대시보드는 2017년 이후의 데이터를 기반으로 미국 주식의 밸류에이션과 성과를 분석합니다.
        
        **주요 기능:**
        1. **Sector Analysis:** SPY, QQQ 및 주요 섹터 ETF의 수익률 비교 (Base 100)
        2. **Valuation Master:**
           * Historical PER 기반 적정주가 시뮬레이션 (2017~2025)
           * Yahoo Finance 예측치(Current/Next Q) 자동 반영
           * PER Band 및 PEG 분석
        3. **Comparison:** 여러 종목의 EPS 성장률 및 PER 추세 상대비교
        
        **주의사항:**
        * 모든 데이터는 실시간이 아니며 지연될 수 있습니다.
        * ChoiceStock의 데이터 구조 변경 시 작동하지 않을 수 있습니다.
        """)
        
    elif menu == "Sector/Market Analysis":
        render_sector_analysis()
        
    elif menu == "Single Stock Valuation":
        render_valuation_analysis()
        
    elif menu == "Comparison Tool":
        render_comparison_analysis()

if __name__ == "__main__":
    main()
