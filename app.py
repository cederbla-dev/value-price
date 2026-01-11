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
        return val

# -----------------------------------------------------------
# [Module 1] 개별 종목 밸류에이션 (File #6: 연도별 적정주가 시뮬레이션)
# -----------------------------------------------------------
def run_single_valuation():
    st.header("💎 개별 종목 밸류에이션 (연도별 적정주가 시뮬레이션)")
    
    # 1. UI 입력
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ticker = st.text_input("티커 입력 (예: TSLA)", "TSLA").upper().strip()
    with col2:
        # 그래프로 보고 싶은 기준 연도 선택
        base_year_input = st.selectbox("차트 기준 연도 (Base Year)", range(2017, 2026), index=0)
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
            eps_df = eps_df.dropna()

            # 회계 분기 매칭을 위한 날짜 보정 (5일 이하는 전월 귀속)
            def adjust_date(dt):
                return (dt.replace(day=1) - timedelta(days=1)).strftime('%Y-%m') if dt.day <= 5 else dt.strftime('%Y-%m')
            
            eps_df.index = [adjust_date(d) for d in eps_df.index]
            eps_df['EPS'] = pd.to_numeric(eps_df['EPS'].astype(str).str.replace(',', ''), errors='coerce')

            # --- [B] Yahoo Finance에서 월간 주가 수집 ---
            stock = yf.Ticker(ticker)
            price_df = stock.history(start="2017-01-01", interval="1mo", auto_adjust=False)
            
            if price_df.empty:
                st.error("주가 데이터를 가져올 수 없습니다."); return
                
            price_df.index = price_df.index.tz_localize(None).strftime('%Y-%m')
            price_df = price_df[['Close']].copy()
            # 월말 데이터 유지를 위해 중복 제거 시 마지막 값 사용
            price_df = price_df[~price_df.index.duplicated(keep='last')]

            # --- [C] 데이터 병합 (EPS + Price) ---
            combined = pd.merge(eps_df, price_df, left_index=True, right_index=True, how='inner')
            combined = combined.sort_index(ascending=True)
            
            if combined.empty:
                st.warning("EPS와 주가 데이터의 날짜가 일치하는 구간이 없습니다."); return

            # --- [D] 미래 예측치(Estimates) 추가 로직 ---
            current_price = price_df['Close'].iloc[-1]
            if include_est != "None":
                est = stock.earnings_estimate
                if est is not None and not est.empty:
                    last_date_obj = pd.to_datetime(combined.index[-1])
                    
                    # Current Q
                    try:
                        curr_val = est['avg'].iloc[0]
                        date_curr = (last_date_obj + pd.DateOffset(months=3)).strftime('%Y-%m')
                        combined.loc[f"{date_curr} (Est.)"] = [curr_val, current_price]
                        
                        # Next Q
                        if include_est == "Next Q" and len(est) > 1:
                            next_val = est['avg'].iloc[1]
                            date_next = (last_date_obj + pd.DateOffset(months=6)).strftime('%Y-%m')
                            combined.loc[f"{date_next} (Est.)"] = [next_val, current_price]
                    except:
                        pass # 예측치 인덱싱 에러 무시

            # --- [E] 연도별 시뮬레이션 및 요약 데이터 생성 ---
            summary_data = []
            final_price = combined['Close'].iloc[-1]
            target_date_label = combined.index[-1]
            
            # 그래프를 그리기 위한 데이터 저장소
            selected_plot_data = None
            selected_scale_factor = 0

            for base_year in range(2017, 2026):
                # 해당 연도 이후 데이터 필터링
                df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                
                if len(df_plot) < 2: continue
                
                base_eps = df_plot.iloc[0]['EPS']
                base_price = df_plot.iloc[0]['Close']
                
                # 적자가 아닌 경우만 분석
                if base_eps <= 0: continue

                # PER 배수 산출
                scale_factor = base_price / base_eps
                # 적정주가(Fair Value) 계산
                df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor

                final_fair_value = df_plot.iloc[-1]['Fair_Value']
                # 괴리율: (현재가 - 적정가) / 적정가
                gap_pct = ((final_price - final_fair_value) / final_fair_value) * 100
                status = "고평가 (Sell)" if gap_pct > 0 else "저평가 (Buy)"

                summary_data.append({
                    "기준 연도": base_year,
                    "적용 PER": scale_factor,
                    "적정 주가": final_fair_value,
                    "현재 주가": final_price,
                    "괴리율 (%)": gap_pct,
                    "판단": status
                })

                # 사용자가 선택한 연도의 데이터 저장
                if base_year == base_year_input:
                    selected_plot_data = df_plot
                    selected_scale_factor = scale_factor

            # --- [F] 결과 시각화 및 출력 ---
            
            # 1. 그래프 출력 (선택한 기준 연도)
            if selected_plot_data is not None:
                fig, ax = plt.subplots(figsize=(14, 7))
                
                # 시장가
                ax.plot(selected_plot_data.index, selected_plot_data['Close'], 
                        label=f'Market Price', color='#1f77b4', linewidth=3, marker='o')
                
                # 적정가
                ax.plot(selected_plot_data.index, selected_plot_data['Fair_Value'], 
                        label=f'Fair Value (Base {base_year_input}, PER {selected_scale_factor:.2f}x)', 
                        color='#d62728', linestyle='--', linewidth=2, marker='s')

                # 예측 구간 하이라이트
                est_indices = [i for i, idx in enumerate(selected_plot_data.index) if "(Est.)" in idx]
                if est_indices:
                    for i in est_indices:
                        ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.2)
                        ax.text(i, selected_plot_data['Fair_Value'].iloc[i], 'Est.', 
                                ha='center', va='bottom', color='red', fontweight='bold')

                ax.set_title(f"[{ticker}] Price vs Fair Value (Base Year: {base_year_input})", fontsize=16)
                ax.legend(fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
            else:
                st.warning(f"{base_year_input}년 기준 데이터가 부족하거나 적자여서 그래프를 그릴 수 없습니다.")

            # 2. 요약 테이블 출력
            if summary_data:
                st.subheader(f"📊 연도별 밸류에이션 시뮬레이션 결과 ({target_date_label} 기준)")
                summary_df = pd.DataFrame(summary_data)
                
                # 소수점 2자리 포맷팅 적용
                summary_df["적용 PER"] = summary_df["적용 PER"].map('{:.2f}'.format)
                summary_df["적정 주가"] = summary_df["적정 주가"].map('{:.2f}'.format)
                summary_df["현재 주가"] = summary_df["현재 주가"].map('{:.2f}'.format)
                summary_df["괴리율 (%)"] = summary_df["괴리율 (%)"].map('{:.2f}'.format)
                
                st.table(summary_df)
                st.info("Tip: '저평가' 신호가 많은 연도가 많을수록 역사적 밸류에이션 하단에 근접했을 확률이 높습니다.")

        except Exception as e:
            st.error(f"데이터 분석 중 오류가 발생했습니다: {e}")

# -----------------------------------------------------------
# [Module 2] 종목 비교 분석 (Files #9, #13: Sync & Comparison)
# -----------------------------------------------------------
# 날짜 정규화 함수 (Shared)
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
                # TTM EPS 추정 (최근 3분기 + 예상 1분기)
                ttm_eps_q1 = sum(historical_eps[-3:]) + est.loc['0q', 'avg']
                combined.loc[q1_dt, 'PER'] = current_price / ttm_eps_q1 if ttm_eps_q1 != 0 else 0
                combined.loc[q1_dt, 'EPS'] = ttm_eps_q1 # 시각화를 위해 저장
                
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
            # 기준점(100) 정규화
            indexed_df = (master_df / master_df.iloc[0]) * 100
            
            fig, ax = plt.subplots(figsize=(15, 8))
            x_labels = [f"{str(d.year)[2:]}Q{d.quarter}" for d in indexed_df.index]
            
            for ticker in indexed_df.columns:
                series = indexed_df[ticker].dropna()
                valid_indices = [indexed_df.index.get_loc(dt) for dt in series.index]
                
                # 예측치 구간 처리
                forecast_count = (1 if q1 else 0) + (1 if q2 else 0)
                
                # 범례에 마지막 값 표시 (소수점 2자리)
                last_val = series.iloc[-1]
                label_txt = f"{ticker} ({last_val:.2f})"

                if forecast_count > 0 and len(valid_indices) > forecast_count:
                    # 실적 구간
                    ax.plot(valid_indices[:-forecast_count], series.values[:-forecast_count], marker='o', label=label_txt)
                    # 예측 구간 (점선)
                    ax.plot(valid_indices[-forecast_count-1:], series.values[-forecast_count-1:], ls='--', marker='x', alpha=0.7)
                else:
                    ax.plot(valid_indices, series.values, marker='o', label=label_txt)
            
            ax.set_xticks(range(len(indexed_df)))
            ax.set_xticklabels(x_labels, rotation=45)
            ax.axhline(100, color='black', alpha=0.5, ls='--')
            ax.set_title(f"Compare: {comp_mode} (Base 100)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("유효한 데이터가 없습니다.")

# -----------------------------------------------------------
# [Module 3] 섹터 수익률 분석 (File #12: Performance)
# -----------------------------------------------------------
def run_sector_perf():
    st.header("📊 섹터 및 지수 수익률 분석 (분기 기준)")
    
    all_tickers = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"]
    selected = st.multiselect("분석할 ETF 선택", all_tickers, default=["SPY", "QQQ", "XLK"])
    
    col1, col2 = st.columns(2)
    with col1:
        sel_year = st.selectbox("시작 연도", range(2017, datetime.now().year + 1))
    with col2:
        sel_quarter = st.selectbox("시작 분기", [1, 2, 3, 4])
    
    q_map = {1: "-01-01", 2: "-04-01", 3: "-07-01", 4: "-10-01"}
    start_date_str = f"{sel_year}{q_map[sel_quarter]}"

    if st.button("수익률 차트 생성"):
        combined_price = pd.DataFrame()
        for t in selected:
            # 수정 주가 (배당 재투자 가정)
            df = yf.Ticker(t).history(start="2017-01-01", interval="1mo", auto_adjust=True)
            if not df.empty:
                df.index = df.index.strftime('%Y-%m-%d')
                combined_price[t] = df['Close']
        
        if not combined_price.empty:
            available_dates = combined_price.index[combined_price.index >= start_date_str]
            if len(available_dates) == 0:
                st.error("해당 시점 이후의 데이터가 없습니다."); return
            
            base_date = available_dates[0]
            # 정규화 (Base=100)
            norm_df = (combined_price.loc[base_date:] / combined_price.loc[base_date]) * 100
            
            fig, ax = plt.subplots(figsize=(15, 8))
            last_val_idx = norm_df.iloc[-1].sort_values(ascending=False)
            
            for ticker in last_val_idx.index:
                lw = 4 if ticker in ["SPY", "QQQ"] else 2
                zo = 5 if ticker in ["SPY", "QQQ"] else 2
                # 범례 소수점 2자리
                ax.plot(norm_df.index, norm_df[ticker], label=f"{ticker} ({last_val_idx[ticker]:.2f})", linewidth=lw, zorder=zo)
            
            # X축 틱 설정
            q_ticks = [d for d in norm_df.index if d.endswith(('-01-01', '-04-01', '-07-01', '-10-01'))]
            ax.set_xticks(q_ticks if q_ticks else norm_df.index[::3])
            plt.xticks(rotation=45)
            ax.axhline(100, color='black', ls='--')
            ax.set_title(f"ETF Performance (Base: {base_date} = 100)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
            
            # --- [Table] 누적 수익률 (%) 표시 ---
            st.write(f"### 🏆 {base_date} 이후 누적 수익률 (%)")
            
            # 100을 빼서 순수 수익률 계산
            performance_pct = (last_val_idx - 100).to_frame(name="수익률 (%)")
            
            # 소수점 2자리 강제 문자열 포맷팅
            performance_pct["수익률 (%)"] = performance_pct["수익률 (%)"].map('{:.2f}'.format)
            
            st.table(performance_pct)

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
        이 시스템은 **ChoiceStock**의 재무 데이터와 **Yahoo Finance**의 시장 데이터를 결합하여 
        다각도로 미국 주식을 분석합니다.
        
        #### 📌 주요 기능
        1. **개별 종목 밸류에이션**: 과거 특정 연도의 PER을 기준으로 현재 주가의 적정성을 시뮬레이션합니다. (File #6)
        2. **종목 비교 분석**: 여러 종목의 PER 및 EPS 성장 추세를 동일한 분기 기준으로 비교합니다. (File #9, #13)
        3. **섹터 수익률**: 주요 ETF 및 지수의 분기별 누적 수익률을 비교합니다. (File #12)
        
        *모든 수치는 소수점 두 자리로 정밀하게 표시됩니다.*
        """)
        
    elif menu == "개별 종목 밸류에이션":
        run_single_valuation()
    elif menu == "종목 비교 분석":
        run_comparison()
    elif menu == "섹터/지수 수익률":
        run_sector_perf()

if __name__ == "__main__":
    main()
