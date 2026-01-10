import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings

# --- 설정 및 경고 무시 ---
warnings.filterwarnings("ignore")
st.set_page_config(page_title="미국주식 통합 분석 시스템 V3", layout="wide")

# --- [로직 통합] 데이터 수집 및 처리 함수 ---

def get_full_data(ticker, base_year, include_curr, include_next):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # 1. 초이스스탁 실적 수집 (파일 1, 3, 7번 로직)
    url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(io.StringIO(res.text))
        
        eps_raw = None
        for df in dfs:
            if df.iloc[:, 0].astype(str).str.contains('EPS').any():
                eps_raw = df.set_index(df.columns[0]).filter(like='EPS', axis=0).transpose()
                break
        
        if eps_raw is None: return None

        # 날짜 보정 로직 (파일 1번: 5일 이하 데이터 전월 마감 처리)
        def adjust_date(dt_str):
            dt = pd.to_datetime(dt_str, format='%y.%m.%d', errors='coerce')
            if pd.isna(dt): return None
            if dt.day <= 5:
                return (dt.replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
            return dt.strftime('%Y-%m')

        eps_raw.index = [adjust_date(i) for i in eps_raw.index]
        for col in eps_raw.columns:
            eps_raw[col] = pd.to_numeric(eps_raw[col].astype(str).str.replace(',', ''), errors='coerce')
        eps_raw = eps_raw.sort_index()

        # 2. 야후 파이낸스 주가 및 예측치 수집 (파일 2, 5, 11번 로직)
        stock = yf.Ticker(ticker)
        price_df = stock.history(start="2017-01-01", interval="1mo")
        price_df.index = price_df.index.strftime('%Y-%m')
        curr_price = stock.fast_info['last_price']
        
        estimates = None
        try:
            est_df = stock.earnings_estimate
            if est_df is not None and not est_df.empty:
                estimates = {'curr': est_df['avg'].iloc[0], 'next': est_df['avg'].iloc[1]}
        except: pass

        return eps_raw, price_df, curr_price, estimates
    except:
        return None

# --- UI 레이아웃 ---
st.title("📊 미국주식 통합 프로 분석 대시보드")
st.markdown("13개의 분석 알고리즘이 결합된 최종 결과물입니다.")

with st.sidebar:
    st.header("🔍 분석 조건")
    ticker = st.text_input("종목 티커", value="AAPL").upper().strip()
    base_year = st.slider("기준 연도 (지수화 기준)", 2017, 2024, 2018)
    
    st.subheader("예측치 반영 설정")
    inc_curr = st.checkbox("현재 분기(Current Qtr) 예측 포함", value=True)
    inc_next = st.checkbox("다음 분기(Next Qtr) 예측 포함", value=False)
    
    run = st.button("분석 실행")

if run:
    with st.spinner('모든 로직을 결합하여 데이터를 산출 중입니다...'):
        result = get_full_data(ticker, base_year, inc_curr, inc_next)
        
        if result:
            eps_df, price_df, curr_price, est = result
            
            # --- [섹션 1] 주가 vs 실적 컨버전스 (파일 4, 5, 6번) ---
            st.subheader("1. 주가 및 EPS 컨버전스 분석")
            
            # 데이터 병합
            combined = pd.merge(eps_df.iloc[:,0], price_df['Close'], left_index=True, right_index=True, how='inner')
            combined.columns = ['EPS', 'Price']
            
            # 지수화(Scaling)
            base_date = f"{base_year}-01"
            # 기준일이 없으면 가장 가까운 날짜 탐색
            if base_date not in combined.index:
                base_date = combined.index[combined.index >= base_date][0]
                
            b_eps = combined.loc[base_date, 'EPS']
            b_price = combined.loc[base_date, 'Price']
            mult = b_price / b_eps
            combined['Fair_Value'] = combined['EPS'] * mult

            # 예측치 추가 로직 (파일 5번)
            plot_df = combined.copy()
            if inc_curr and est:
                new_idx = (datetime.now() + timedelta(days=45)).strftime('%Y-%m')
                plot_df.loc[new_idx + "(E)"] = [est['curr'], np.nan, est['curr'] * mult]
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Price'], name="실제 주가", line=dict(color='blue', width=3)))
            fig1.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Fair_Value'], name="EPS 기반 적정가", line=dict(color='red', dash='dash')))
            st.plotly_chart(fig1, use_container_width=True)

            # --- [섹션 2] 밸류에이션 요약 테이블 (파일 6, 10번) ---
            st.divider()
            st.subheader("2. 연도별 적정가 및 괴리율 요약")
            
            summary = []
            for idx in combined.index[-5:]: # 최근 5개 분기
                fv = combined.loc[idx, 'Fair_Value']
                pr = combined.loc[idx, 'Price']
                gap = ((pr / fv) - 1) * 100
                summary.append({
                    "날짜": idx,
                    "EPS": f"{combined.loc[idx, 'EPS']:.2f}",
                    "실제 주가": f"${pr:.2f}",
                    "적정 가치": f"${fv:.2f}",
                    "상태": "고평가" if gap > 0 else "저평가",
                    "괴리율": f"{gap:+.2f}%"
                })
            st.table(pd.DataFrame(summary))

            # --- [섹션 3] PER / PEG 정밀 분석 (파일 8, 9, 11번) ---
            st.divider()
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("3. PER 추이 및 평균/중위값")
                # TTM PER 계산 (파일 7번 로직)
                combined['TTM_EPS'] = combined['EPS'].rolling(window=4).sum()
                combined['PER'] = combined['Price'] / combined['TTM_EPS']
                per_data = combined['PER'].dropna()
                
                avg_per = per_data.mean()
                med_per = per_data.median()
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=per_data.index, y=per_data.values, name="PER"))
                fig2.add_hline(y=avg_per, line_color="red", line_dash="dot", annotation_text=f"평균:{avg_per:.1f}")
                fig2.add_hline(y=med_per, line_color="green", line_dash="dash", annotation_text=f"중위:{med_per:.1f}")
                st.plotly_chart(fig2, use_container_width=True)

            with c2:
                st.subheader("4. Forward PEG 분석")
                if est:
                    # 파일 11번 PEG 로직 적용
                    last_ttm = combined['TTM_EPS'].iloc[-1]
                    future_growth = ((est['curr'] * 4 / last_ttm) - 1) * 100
                    curr_per = curr_price / last_ttm
                    peg = curr_per / future_growth if future_growth > 0 else 0
                    
                    st.metric("현재 TTM PER", f"{curr_per:.2f}x")
                    st.metric("예상 성장률(G)", f"{future_growth:+.2f}%")
                    st.metric("최종 PEG Now", f"{peg:.2f}")
                    
                    if 0 < peg < 1: st.success("성장성 대비 저평가 (PEG < 1)")
                    else: st.warning("성장성 수치 확인 필요 (PEG > 1)")

            # --- [섹션 4] 섹터 및 수익률 비교 (파일 12, 13번) ---
            st.divider()
            st.subheader("5. 섹터 벤치마크 대비 수익률 비교 (지수 100 기준)")
            benchmarks = ["SPY", "QQQ", ticker]
            b_data = yf.download(benchmarks, start=f"{base_year}-01-01")['Close']
            b_norm = (b_data / b_data.iloc[0]) * 100
            
            fig3 = go.Figure()
            for col in b_norm.columns:
                fig3.add_trace(go.Scatter(x=b_norm.index, y=b_norm[col], name=col))
            st.plotly_chart(fig3, use_container_width=True)
            
        else:
            st.error("데이터를 불러오지 못했습니다. 티커를 확인해 주세요.")
