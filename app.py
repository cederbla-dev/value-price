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

# ✅ AgGrid 추가 (유일한 신규 import)
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# 기본 설정
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock & ETF Professional Analyzer", layout="wide")

# session_state 초기화
if "v3_run" not in st.session_state:
    st.session_state.v3_run = False

if "v3_select_mode" not in st.session_state:
    st.session_state.v3_select_mode = False

if "v3_selected_values" not in st.session_state:
    st.session_state.v3_selected_values = []

if "v3_per_avg" not in st.session_state:
    st.session_state.v3_per_avg = None

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

                    # 그래프 시각화 설정 (수정됨: 사이즈 축소 9.6, 4.8)
                    fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor='white')
                    
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
                    
                    # --- [범례 커스텀 수정] 위치 외부로 변경 (bbox_to_anchor) ---
                    leg = ax.legend(
                        loc='upper left', 
                        bbox_to_anchor=(1.02, 1), # 그래프 밖 우측 상단으로 이동
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
            v2_ticker = st.text_input("🏢 분석 티커 입력", "AAPL").upper().strip()
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


# --- 메뉴 3: 개별종목 적정주가 분석 3 (PER 테이블 AgGrid 적용) ---

if main_menu == "개별종목 적정주가 분석 3":

    with st.container(border=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            v3_ticker = st.text_input("🏢 티커 입력", "MSFT").upper().strip()
        with col2:
            v3_start_year = st.number_input("📅 기준 연도", 2010, 2025, 2017)

        v3_selected_metric = st.radio(
            "📈 분석 지표 선택",
            ("PER 그래프", "PER 테이블"),
            horizontal=True
        )

        if st.button("데이터 분석 실행", type="primary", use_container_width=True):
            st.session_state.v3_run = True

    # ==================================================
    # 데이터 분석 실행 상태
    # ==================================================
    if st.session_state.v3_run and v3_ticker:

        url = f"https://www.choicestock.co.kr/search/invest/{v3_ticker}/MRQ"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        dfs = pd.read_html(io.StringIO(response.text))

        target_df = next(
            (df.set_index(df.columns[0]) for df in dfs
             if df.iloc[:, 0].astype(str).str.contains('PER|EPS').any()),
            None
        )

        per_raw = target_df[target_df.index.astype(str).str.contains('PER')].transpose()
        eps_raw = target_df[target_df.index.astype(str).str.contains('EPS')].transpose()

        combined = pd.DataFrame({
            'PER': pd.to_numeric(per_raw.iloc[:, 0], errors='coerce'),
            'EPS': pd.to_numeric(eps_raw.iloc[:, 0].astype(str).str.replace(',', ''), errors='coerce')
        }).dropna()

        combined.index = pd.to_datetime(combined.index, format='%y.%m.%d')
        combined = combined.sort_index()

        def q_label(dt):
            adj = dt if dt.day > 5 else (dt - timedelta(days=5))
            return f"{str(adj.year)[2:]}.Q{(adj.month - 1)//3 + 1}"

        combined['Label'] = combined.index.map(q_label)
        plot_df = combined[combined.index >= f"{v3_start_year}-01-01"].copy()

        # ==================================================
        # PER 테이블 + 버튼 UX
        # ==================================================
        if v3_selected_metric == "PER 테이블":

            st.subheader("📊 연도 / 분기별 PER")

            table_df = plot_df[['Label', 'PER']].copy()
            table_df['Year'] = table_df['Label'].apply(lambda x: "20" + x.split('.')[0])
            table_df['Quarter'] = table_df['Label'].apply(lambda x: x.split('.')[1])

            pivot_df = table_df.pivot(index='Year', columns='Quarter', values='PER')
            pivot_df = pivot_df.reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'])
            pivot_df = pivot_df.reset_index()

            # -----------------------------
            # 버튼 영역
            # -----------------------------
            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("셀 선택 시작"):
                    st.session_state.v3_select_mode = True
                    st.session_state.v3_selected_values = []
                    st.session_state.v3_per_avg = None

            with c2:
                if st.button("평균값 계산"):
                    if st.session_state.v3_selected_values:
                        st.session_state.v3_per_avg = (
                            sum(st.session_state.v3_selected_values)
                            / len(st.session_state.v3_selected_values)
                        )

            with c3:
                if st.button("적정주가 계산"):
                    if st.session_state.v3_per_avg:
                        eps_latest = plot_df['EPS'].iloc[-1]
                        fair_price = st.session_state.v3_per_avg * eps_latest
                        st.info(f"📌 적정주가: {fair_price:,.0f}")

            # -----------------------------
            # AgGrid
            # -----------------------------
            gb = GridOptionsBuilder.from_dataframe(pivot_df)
            gb.configure_grid_options(enableRangeSelection=True)
            gb.configure_selection(selection_mode="multiple", use_checkbox=False)

            grid_response = AgGrid(
                pivot_df,
                gridOptions=gb.build(),
                update_mode=GridUpdateMode.MODEL_CHANGED,
                height=320,
                fit_columns_on_grid_load=True
            )

            # -----------------------------
            # 선택 모드일 때만 값 누적
            # -----------------------------
            if st.session_state.v3_select_mode:
                for cell in grid_response.get("selected_cells", []):
                    val = cell.get("value")
                    if isinstance(val, (int, float)):
                        if val not in st.session_state.v3_selected_values:
                            st.session_state.v3_selected_values.append(val)

            # -----------------------------
            # 상태 출력
            # -----------------------------
            st.write(
                f"선택된 셀 수: {len(st.session_state.v3_selected_values)} | "
                f"누적 PER 합계: {sum(st.session_state.v3_selected_values):.2f}"
            )

            if st.session_state.v3_per_avg:
                st.success(f"📊 평균 PER: {st.session_state.v3_per_avg:.2f}")



# --- 메뉴 4: 개별종목 적정주가 분석 4 ---
elif main_menu == "개별종목 적정주가 분석 4":
    with st.container(border=True):
        v4_ticker = st.text_input("🏢 분석 티커 입력 (PEG 분석)", "AAPL").upper().strip()
        run_v4 = st.button("연도별 정밀 PEG 분석 실행", type="primary", use_container_width=True)

    if run_v4 and v4_ticker:
        try:
            with st.spinner(f"[{v4_ticker}] 데이터 수집 및 연도별 정밀 분석 중..."):
                # 1. 초이스스탁 EPS 데이터 수집
                url = f"https://www.choicestock.co.kr/search/invest/{v4_ticker}/MRQ"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                
                resp = requests.get(url, headers=headers, timeout=10)
                dfs = pd.read_html(io.StringIO(resp.text))
                
                target_df = next((df for df in dfs if df.iloc[:, 0].astype(str).str.contains('EPS', na=False).any()), None)
                
                if target_df is None:
                    st.error("⚠️ 해당 종목의 분기별 EPS 데이터를 찾을 수 없습니다.")
                else:
                    # 2. 데이터 전처리
                    target_df = target_df.set_index(target_df.columns[0])
                    eps_df = target_df[target_df.index.str.contains('EPS', na=False)].transpose()
                    eps_df.index = pd.to_datetime(eps_df.index, format='%y.%m.%d', errors='coerce')
                    eps_df = eps_df.dropna().sort_index()
                    eps_df.columns = ['Quarterly_EPS']
                    
                    # 3. 주가 및 야후 파이낸스 추정치 수집 (방어적 코드)
                    stock = yf.Ticker(v4_ticker)
                    hist = stock.history(period="5d")
                    if hist.empty:
                        st.error("⚠️ 주가 데이터를 가져올 수 없습니다. 티커를 확인하세요.")
                        st.stop()
                    
                    current_price = hist['Close'].iloc[-1]
                    
                    # 추정치 데이터 확보 (yfinance 최신버전 호환용)
                    try:
                        estimates = stock.earnings_estimate
                        if estimates is None or estimates.empty:
                            # 대안: info에서 forwardEps 가져오기
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

                    # 4. 분석 변수 설정
                    latest_date = eps_df.index[-1]
                    latest_month = latest_date.month
                    latest_idx = len(eps_df) - 1

                    def get_ttm(idx):
                        if idx < 3: return None
                        return eps_df['Quarterly_EPS'].iloc[idx-3 : idx+1].sum()

                    # --- 분석 로직 분기 실행 ---
                    results = []
                    analysis_type = ""
                    base_date = latest_date

                    # A. 확정 실적 기준 (10, 11, 12월 마감)
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

                    # B. 미래 1Q 포함 (7, 8, 9월 마감)
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

                    # C. 미래 2Q 포함 (4, 5, 6월 마감)
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

                    # D. 연초 데이터 부족 (1, 2, 3월 마감)
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

                    # 5. 결과 출력
                    if results:
                        st.subheader(f"📌 {analysis_type}")
                        st.caption(f"기준일: {base_date.strftime('%Y-%m-%d')} | 현재가: ${current_price:.2f}")
                        
                        df_res = pd.DataFrame(results)
                        df_res.columns = ['분석 기간', '과거 TTM EPS', '기준 TTM EPS', '연평균성장률(%)', 'PER', 'PEG']
                        
                        # 스타일링 및 테이블 출력
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
