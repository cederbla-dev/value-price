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

# --- 메뉴 1: 개별종목 적정주가 분석 1 (범례 배경색 및 정렬 최종 수정본) ---
if main_menu == "개별종목 적정주가 분석 1":
    # 1. 상단 입력 UI 레이아웃
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
            # 데이터 가져오기 (기존 정의된 fetch_valuation_data 함수 호출)
            combined = fetch_valuation_data(val_ticker, val_predict_mode)
            
            if combined is not None and not combined.empty:
                final_price = combined['Close'].iloc[-1]
                target_date_label = combined.index[-1]
                summary_list = []

                # --- 파트 A: 연도별 그래프 생성 ---
                st.subheader(f"📈 {val_ticker} 연도별 적정주가 시뮬레이션")
                
                for base_year in range(2017, 2026):
                    df_plot = combined[combined.index >= f'{base_year}-01'].copy()
                    
                    if len(df_plot) < 2 or df_plot.iloc[0]['EPS'] <= 0:
                        continue
                    
                    # 기준 PER 산출 및 적정가(Fair Value) 계산
                    scale_factor = df_plot.iloc[0]['Close'] / df_plot.iloc[0]['EPS']
                    df_plot['Fair_Value'] = df_plot['EPS'] * scale_factor
                    
                    last_fair_value = df_plot.iloc[-1]['Fair_Value']
                    gap_pct = ((final_price - last_fair_value) / last_fair_value) * 100
                    status = "🔴 고평가" if gap_pct > 0 else "🔵 저평가"

                    # 표 데이터 저장용 리스트업
                    summary_list.append({
                        "기준 연도": f"{base_year}년",
                        "기준 PER": f"{scale_factor:.1f}x",
                        "적정 주가": f"${last_fair_value:.2f}",
                        "현재 주가": f"${final_price:.2f}",
                        "괴리율 (%)": f"{gap_pct:+.1f}%",
                        "상태": status
                    })

                    # 그래프 시각화 설정
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
                    
                    # 1. Price 라인 (파란색)
                    ax.plot(df_plot.index, df_plot['Close'], color='#1f77b4', 
                            linewidth=2.0, marker='o', markersize=4, label='Price')
                    # 2. EPS 가치 라인 (빨간색)
                    ax.plot(df_plot.index, df_plot['Fair_Value'], color='#d62728', 
                            linestyle='--', marker='s', markersize=4, label='EPS')
                    
                    # 미래 예측(Est.) 구간 하이라이트
                    for i, idx in enumerate(df_plot.index):
                        if "(Est.)" in str(idx):
                            ax.axvspan(i-0.5, i+0.5, color='orange', alpha=0.1)

                    # 스타일 적용 (기존 apply_strong_style 함수)
                    apply_strong_style(ax, f"Base Year: {base_year} (Gap: {gap_pct:+.1f}%)", "Price ($)")
                    plt.xticks(rotation=45)
                    
                    # --- [범례 커스텀 수정] 배경 흰색 및 글자색 지정 ---
                    leg = ax.legend(
                        loc='upper left', 
                        fontsize=11, 
                        frameon=True, 
                        facecolor='white',  # 범례 내부 배경색 흰색
                        edgecolor='black',  # 범례 테두리색 검정
                        framealpha=1.0      # 투명도 없음 (불투명 흰색)
                    )
                    
                    # 범례 내 텍스트 색상 및 굵기 개별 설정
                    for text in leg.get_texts():
                        if text.get_text() == 'Price':
                            text.set_color('#1f77b4')  # 파란색 글씨
                            text.set_weight('bold')
                        elif text.get_text() == 'EPS':
                            text.set_color('#d62728')  # 빨간색 글씨
                            text.set_weight('bold')
                    
                    st.pyplot(fig)
                    plt.close(fig)

                # --- 파트 B: 최종 요약 표 출력 (60% 너비 및 왼쪽 정렬) ---
                if summary_list:
                    st.write("\n")
                    st.markdown("---")
                    st.subheader(f"📊 {val_ticker} 밸류에이션 종합 요약")
                    st.caption(f"분석 기준점(Target Date): {target_date_label}")

                    summary_df = pd.DataFrame(summary_list)

                    # 표의 시작점을 그래프의 왼쪽 끝과 맞추기 위해 6:4 비율 컬럼 사용
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
        # vertical_alignment="bottom"을 추가하여 입력창과 버튼의 높이를 정렬합니다.
        col1, col2, col3 = st.columns([0.5, 0.5, 1], vertical_alignment="bottom")
        with col1:
            v2_ticker = st.text_input("🏢 분석 티커 입력", "PAYX").upper().strip()
        with col2:
            # 기존의 st.write("") 공백 제거 후 버튼 배치
            run_v2 = st.button("당해 EPS 기반 분석", type="primary", use_container_width=True)
        with col3:
            # 우측 50% 공간 비워둠
            pass

    if run_v2 and v2_ticker:
        try:
            with st.spinner('데이터를 수집하고 분석 중입니다...'):
                stock = yf.Ticker(v2_ticker)
                
                # 1. 과거 실적 수집
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
                
                # 2. 주가 및 예측치 수집
                price_history = stock.history(start="2017-01-01", interval="1d")
                price_df = price_history['Close'].copy()
                if price_df.index.tz is not None:
                    price_df.index = price_df.index.tz_localize(None)
                    
                current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
                estimates = stock.earnings_estimate
                current_q_est = estimates['avg'].iloc[0] if estimates is not None else 0

                # 3. 타겟 EPS 계산
                recent_3_actuals = raw_eps['EPS'].iloc[-3:].sum()
                final_target_eps = recent_3_actuals + current_q_est

                # 4. 4분기 단위 프로세싱
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

                # 5. UI 출력
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

                # 요약 정보
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
            with st.spinner('데이터를 수집하고 미래 가치를 계산 중입니다...'):
                # 1. 과거 데이터 수집 (ChoiceStock)
                url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                dfs = pd.read_html(io.StringIO(response.text))
                
                target_df = next((df.set_index(df.columns[0]) for df in dfs if df.iloc[:, 0].astype(str).str.contains('PER|EPS').any()), None)
                
                if target_df is not None:
                    # 데이터 추출 및 정렬
                    per_raw = target_df[target_df.index.astype(str).str.contains('PER')].transpose()
                    eps_raw = target_df[target_df.index.astype(str).str.contains('EPS')].transpose()
                    
                    combined = pd.DataFrame({
                        'PER': pd.to_numeric(per_raw.iloc[:, 0], errors='coerce'),
                        'EPS': pd.to_numeric(eps_raw.iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
                    }).dropna()
                    
                    combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
                    combined = combined.sort_index()
                    
                    # 라벨 생성 함수 (원본 코드 로직)
                    def get_q_label(dt):
                        year = dt.year if dt.day > 5 else (dt - timedelta(days=5)).year
                        month = dt.month if dt.day > 5 else (dt - timedelta(days=5)).month
                        q = (month-1)//3 + 1
                        return f"{str(year)[2:]}.Q{q}"

                    combined['Label'] = [get_q_label(d) for d in combined.index]
                    plot_df = combined[combined.index >= f"{base_year}-01-01"].copy()

                    # 2. 미래 예측 로직 적용 (원본 코드의 슬라이딩 TTM 엔진)
                    if v3_predict_mode != "None":
                        stock = yf.Ticker(v3_ticker)
                        current_price = stock.fast_info.get('last_price', stock.history(period="1d")['Close'].iloc[-1])
                        est = stock.earnings_estimate
                        
                        if est is not None and not est.empty:
                            historical_eps = combined['EPS'].tolist()
                            last_label = plot_df['Label'].iloc[-1]
                            last_yr = int("20" + last_label.split('.')[0])
                            last_q = int(last_label.split('Q')[1])

                            # 현재 분기 예측 추가
                            curr_q_est = est.loc['0q', 'avg']
                            t1_q, t1_yr = (last_q + 1, last_yr) if last_q < 4 else (1, last_yr + 1)
                            label_1 = f"{str(t1_yr)[2:]}.Q{t1_q}(E)"
                            ttm_eps_1 = sum(historical_eps[-3:]) + curr_q_est
                            per_1 = current_price / ttm_eps_1
                            
                            # 데이터프레임에 추가
                            new_idx1 = pd.Timestamp(f"{t1_yr}-{(t1_q-1)*3+1}-01")
                            plot_df.loc[new_idx1] = [per_1, np.nan, label_1]

                            # 다음 분기 예측 추가
                            if v3_predict_mode == "다음 분기 예측":
                                next_q_est = est.loc['+1q', 'avg']
                                t2_q, t2_yr = (t1_q + 1, t1_yr) if t1_q < 4 else (1, t1_yr + 1)
                                label_2 = f"{str(t2_yr)[2:]}.Q{t2_q}(E)"
                                ttm_eps_2 = sum(historical_eps[-2:]) + curr_q_est + next_q_est
                                per_2 = current_price / ttm_eps_2
                                
                                new_idx2 = pd.Timestamp(f"{t2_yr}-{(t2_q-1)*3+1}-01")
                                plot_df.loc[new_idx2] = [per_2, np.nan, label_2]

                    # 3. 통계치 계산 및 시각화 설정
                    avg_per = plot_df['PER'].mean()
                    max_per = plot_df['PER'].max()
                    min_per = plot_df['PER'].min()
                    x_labels = plot_df['Label'].values
                    
                    fig, ax = plt.subplots(figsize=(11.0, 5.5), facecolor='white')
                    
                    # PER 선 그래프
                    ax.plot(x_labels, plot_df['PER'].values, marker='o', color='#34495e', linewidth=2, zorder=3)
                    
                    # 평균선 (Middle)
                    ax.axhline(avg_per, color='#e74c3c', linestyle='--', linewidth=1.5, zorder=2)
                    
                    # Y축 중앙 정렬 (Middle 기준 상하 대칭)
                    half_range = max(max_per - avg_per, avg_per - min_per) * 1.4
                    ax.set_ylim(avg_per - half_range, avg_per + half_range)

                    # 좌측 상단 범례 직접 표시
                    ax.text(0.02, 0.95, "PER", color='#34495e', fontweight='bold', transform=ax.transAxes, fontsize=11, va='top')
                    ax.text(0.02, 0.88, "Middle", color='#e74c3c', fontweight='bold', transform=ax.transAxes, fontsize=11, va='top')

                    # 미래 예측 구간 하이라이트 및 수치 표시
                    for i, label in enumerate(x_labels):
                        if "(E)" in label:
                            # 옅은 노란색 배경 채우기
                            ax.axvspan(i-0.5, i+0.5, color='#fff9c4', alpha=0.5, zorder=1)
                            # 예측 PER 수치 텍스트 표시
                            ax.text(i, plot_df['PER'].iloc[i] + (half_range*0.1), f"{plot_df['PER'].iloc[i]:.1f}", 
                                    ha='center', fontweight='bold', color='#d35400', fontsize=9)
                            # Forecast 라벨
                            ax.text(i, ax.get_ylim()[0] + (half_range*0.1), "Forecast", 
                                    color='#fbc02d', fontsize=8, ha='center', fontweight='bold')

                    if hasattr(plt, 'apply_strong_style'): # 사용자 정의 스타일 함수가 있을 경우 실행
                        apply_strong_style(ax, f"{v3_ticker} PER Valuation Trend", "PER Ratio")
                    else:
                        ax.set_title(f"{v3_ticker} PER Valuation Trend", fontsize=14, pad=15)
                        ax.set_ylabel("PER Ratio")
                    
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # 하단 요약 정보 박스
                    st.info(f"💡 **분석 요약:** 현재 평균 PER은 **{avg_per:.2f}**이며, 마지막 데이터는 **{plot_df['PER'].iloc[-1]:.2f}**입니다.")
                else:
                    st.warning("데이터를 불러오지 못했습니다. 티커를 확인해 주세요.")
                    
        except Exception as e: 
            st.error(f"분석 중 오류 발생: {e}")


# --- 메뉴 4: 개별종목 적정주가 분석 4 (테이블 너비 20% 확대: 550) ---
elif main_menu == "개별종목 적정주가 분석 4":
    with st.container(border=True):
        v4_ticker = st.text_input("🏢 분석 티커 입력 (PEG 분석)", "AAPL").upper().strip()
        run_v4 = st.button("연도별 정밀 PEG 분석 실행", type="primary", use_container_width=True)

    if run_v4 and v4_ticker:
        try:
            with st.spinner(f"[{v4_ticker}] 연도별 정밀 PEG 분석 중..."):
                url = f"https://www.choicestock.co.kr/search/invest/{v4_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0'}
                dfs = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))
                target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS').any()), None)
                if target_df is not None:
                    target_df = target_df.set_index(target_df.columns[0])
                    eps_df = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                    eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
                    eps_df = eps_df.dropna().sort_index()
                    eps_df.columns = ['Quarterly_EPS']
                    stock = yf.Ticker(v4_ticker)
                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                    estimates = stock.earnings_estimate
                    latest_date = eps_df.index[-1]
                    def get_ttm(idx): return eps_df['Quarterly_EPS'].iloc[idx-3 : idx+1].sum() if idx >= 3 else None

                    def display_peg_table(title, date, data_list):
                        st.subheader(f"📌 {title} (기준일: {date.date()})")
                        df_res = pd.DataFrame(data_list)
                        df_res.columns = ['분석 기간', '과거 TTM EPS', '기준 TTM EPS', '연평균성장률(%)', 'PER', 'PEG']
                        # 너비를 기존 450에서 약 20% 늘린 550으로 설정
                        st.dataframe(df_res.style.format({
                            '과거 TTM EPS': '{:.2f}', '기준 TTM EPS': '{:.2f}',
                            '연평균성장률(%)': '{:.2f}', 'PER': '{:.2f}', 'PEG': '{:.2f}'
                        }), width=550, hide_index=True)

                    results = []
                    per_val = current_price / get_ttm(len(eps_df)-1)
                    for y in range(5, 0, -1):
                        t_idx = len(eps_df)-1 - (y*4)
                        if t_idx >= 3:
                            past_eps, curr_eps = get_ttm(t_idx), get_ttm(len(eps_df)-1)
                            growth = ((curr_eps/past_eps)**(1/y)-1)*100
                            results.append({
                                'period': f"최근 {y}년 연간", 'past': past_eps, 'curr': curr_eps,
                                'growth': growth, 'per': per_val, 'peg': per_val/growth if growth > 0 else 0
                            })
                    display_peg_table("[확정 실적 기준] 연간 PEG", latest_date, results)
        except Exception as e: st.error(f"오류: {e}")

# --- 메뉴 5: 기업 가치 비교 (PER/EPS) ---
elif main_menu == "기업 가치 비교 (PER/EPS)":
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 2])
        ticker_input = col1.text_input("🏢 티커 입력", "AAPL, MSFT, NVDA")
        start_year = col2.number_input("📅 기준 연도", 2010, 2025, 2020)
        predict_mode = col3.radio("🔮 미래 예측 옵션", ("None", "현재 분기 예측", "다음 분기 예측"), horizontal=True)
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
                for ticker in indexed_per.columns:
                    ax.plot(indexed_per.index.strftime('%yQ%q'), indexed_per[ticker], marker='o', label=ticker, linewidth=2)
                apply_strong_style(ax, f"Relative PER Change since {start_year}", "Change (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                plt.xticks(rotation=45); ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                st.pyplot(fig)
        else:
            all_eps = []
            for t in tickers:
                df = fetch_eps_data(t, predict_mode)
                if not df.empty: all_eps.append(df)
            if all_eps:
                fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                for df in all_eps:
                    t = [c for c in df.columns if c != 'type'][0]
                    plot_df = df[df.index >= f"{start_year}-Q1"]
                    norm_vals = (plot_df[t] / plot_df[t].iloc[0] - 1) * 100
                    ax.plot(plot_df.index, norm_vals, marker='o', label=t, linewidth=2)
                apply_strong_style(ax, f"Normalized EPS Growth since {start_year}-Q1", "Growth (%)")
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                plt.xticks(rotation=45); ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
                st.pyplot(fig)

# --- 메뉴 6: ETF 섹터 수익률 분석 ---
else:
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        selected_etfs = col1.multiselect("🌐 ETF 선택", ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "QQQ"], default=["SPY", "QQQ", "XLK", "XLE"])
        start_year_etf = col2.number_input("📅 기준 연도", 2010, 2025, 2020)
        start_q_etf = col3.selectbox("🔢 기준 분기", [1, 2, 3, 4])
        run_etf_btn = st.button("ETF 수익률 분석 시작", type="primary", use_container_width=True)
    if run_etf_btn and selected_etfs:
        df_etf = fetch_etf_data(selected_etfs)
        start_date = f"{start_year_etf}-{str((start_q_etf-1)*3 + 1).zfill(2)}"
        norm_etf = (df_etf.loc[start_date:] / df_etf.loc[start_date:].iloc[0] - 1) * 100
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        for ticker in norm_etf.columns:
            ax.plot(norm_etf.index, norm_etf[ticker], label=ticker, linewidth=2.5 if ticker in ["SPY", "QQQ"] else 1.5)
        apply_strong_style(ax, f"ETF Performance since {start_date}", "Return (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xticks(rotation=45); ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        st.pyplot(fig)
