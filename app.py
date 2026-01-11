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
                
                # 예측치를 담을 리스트 (인덱스 중복 방지 및 순서 보장)
                if predict_mode in ["현재 분기 예측", "다음 분기 예측"]:
                    # 1번째 예측 분기 (0q)
                    q1_year = year + (q // 4)
                    q1_q = (q % 4) + 1
                    label_q1 = f"{q1_year}-Q{q1_q}"
                    eps_df.loc[label_q1, ticker] = est.loc['0q', 'avg']
                    eps_df.loc[label_q1, 'type'] = 'Estimate'
                    
                    if predict_mode == "다음 분기 예측":
                        # 2번째 예측 분기 (+1q)
                        q2_year = q1_year + (q1_q // 4)
                        q2_q = (q1_q % 4) + 1
                        label_q2 = f"{q2_year}-Q{q2_q}"
                        eps_df.loc[label_q2, ticker] = est.loc['+1q', 'avg']
                        eps_df.loc[label_q2, 'type'] = 'Estimate'
        
        return eps_df.sort_index()
    except: return pd.DataFrame()

# --- Tab 2 (EPS) 그래프 출력 부분 수정 ---
with tab2:
    all_eps = []
    for t in tickers:
        df = fetch_eps_data(t, predict_mode)
        if not df.empty: all_eps.append(df)
    
    if all_eps:
        # 전체 인덱스 통합 및 정렬
        full_idx = sorted(list(set().union(*(d.index for d in all_eps))))
        filtered_idx = [idx for idx in full_idx if idx >= f"{start_year}-Q1"]
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        for i, df in enumerate(all_eps):
            t = [c for c in df.columns if c != 'type'][0]
            plot_df = df.reindex(filtered_idx)
            
            # 기준값 설정 (정규화용)
            base_data = plot_df[plot_df[t].notna()]
            if base_data.empty: continue
            base_val = base_data[t].iloc[0]
            
            norm_vals = (plot_df[t] / base_val - 1) * 100
            color = plt.cm.Set1(i % 9)
            
            # 실선(Actual)과 점선(Estimate) 구분하여 그리기
            # 실제 값의 마지막 위치 찾기
            actual_indices = np.where(plot_df['type'] == 'Actual')[0]
            if len(actual_indices) > 0:
                last_act_idx = actual_indices[-1]
                
                # 1. 실선 (처음부터 마지막 Actual까지)
                ax.plot(range(last_act_idx + 1), norm_vals.iloc[:last_act_idx + 1], 
                        marker='o', label=f"{t} ({norm_vals.dropna().values[-1]:+.1f}%)", 
                        color=color, linewidth=2.5)
                
                # 2. 점선 (마지막 Actual부터 끝까지)
                if predict_mode != "None":
                    ax.plot(range(last_act_idx, len(filtered_idx)), norm_vals.iloc[last_act_idx:], 
                            linestyle='--', color=color, linewidth=2)
                    # 예측 지점 강조 (다이아몬드 마커)
                    est_indices = np.where(plot_df['type'] == 'Estimate')[0]
                    ax.scatter(est_indices, norm_vals.iloc[est_indices], 
                               marker='D', s=50, color=color, zorder=5)

        apply_strong_style(ax, f"Normalized EPS Growth (%) since {start_year}-Q1", "Growth (%)")
        ax.set_xticks(range(len(filtered_idx)))
        ax.set_xticklabels(filtered_idx, rotation=45)
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', labelcolor='black')
        st.pyplot(fig)
