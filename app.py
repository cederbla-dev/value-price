import requests
import pandas as pd
import yfinance as yf
import io
from datetime import datetime
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def analyze_calendar_final_with_pct():
    ticker = input("분석할 티커를 입력하세요 (예: PAYX, AAPL): ").upper().strip()
    if not ticker: return

    print(f"\n[{ticker}] 데이터 정제 및 괴리율 포함 밸류에이션 생성 중...")

    try:
        stock = yf.Ticker(ticker)
        
        # 1. 과거 실적 데이터 수집
        url = f"https://www.choicestock.co.kr/search/invest/{ticker}/MRQ"
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

        # 2. 주가 데이터 수집 및 시간대 제거
        price_history = stock.history(start="2017-01-01", interval="1d")
        price_df = price_history['Close'].copy()
        if price_df.index.tz is not None:
            price_df.index = price_df.index.tz_localize(None)
            
        current_price = stock.fast_info.get('last_price', price_df.iloc[-1])
        
        # 야후 Analysis 예측치
        estimates = stock.earnings_estimate
        current_q_est = estimates['avg'].iloc[0] if estimates is not None else 0

        # 3. 타겟 EPS 계산 (최근 확정 3개 + 예측 1개 = 4.65)
        recent_3_actuals = raw_eps['EPS'].iloc[-3:].sum()
        final_target_eps = recent_3_actuals + current_q_est

        # 4. 발표일 기준 4개씩 묶기
        processed_data = []
        for i in range(0, len(raw_eps) - 3, 4):
            group = raw_eps.iloc[i:i+4]
            eps_sum = group['EPS'].sum()
            start_date = group.index[0]
            end_date = group.index[-1]
            
            avg_price = price_df[start_date:end_date].mean()
            year_label = f"{start_date.year}년"
            
            is_last_row = (i + 4 >= len(raw_eps))
            eps_display = f"{eps_sum:.2f}"
            
            if is_last_row:
                eps_display = f"{final_target_eps:.2f}(예상치)"
                eps_sum = final_target_eps
            
            processed_data.append({
                'Period': year_label,
                'EPS_Display': eps_display,
                'EPS_Val': eps_sum,
                'Avg_Price': avg_price,
                'PER': avg_price / eps_sum if eps_sum > 0 else 0
            })

        # 5. 결과 출력
        print("\n" + "="*115)
        print(f" [{ticker}] 회계 연도 무시 / 실제 발표일 기준 4분기 단위 과거 밸류에이션 기록")
        print(f" * 분석 기준 EPS (최근 3개 확정 + 1개 예측): ${final_target_eps:.2f}")
        print("="*115)
        # 괴리율 항목 추가
        print(f"{'기준 연도':^15} | {'4분기 EPS합':^18} | {'평균 주가':^12} | {'평균 PER':^12} | {'적정주가 가치':^12} | {'현재가 판단'}")
        print("-" * 115)

        past_pers = [d['PER'] for d in processed_data if d['PER'] > 0]
        avg_past_per = np.mean(past_pers) if past_pers else 0

        for data in processed_data:
            fair_price = final_target_eps * data['PER']
            diff_pct = ((current_price / fair_price) - 1) * 100
            status = "저평가" if current_price < fair_price else "고평가"
            
            # 괴리율 수치 포함하여 출력
            print(f"{data['Period']:^17} | {data['EPS_Display']:^20} | {data['Avg_Price']:^14.2f} | {data['PER']:^14.1f} | {fair_price:^14.2f} | {abs(diff_pct):>6.1f}% {status}")

        # 6. 현재 상황 별도 표시
        current_fair_value = final_target_eps * avg_past_per
        current_diff = ((current_price / current_fair_value) - 1) * 100
        current_status = "저평가" if current_price < current_fair_value else "고평가"

        print("="*115)
        print(f" [조회 시점 현재 상황 요약]")
        print(f" > 현재 실시간 주가: ${current_price:.2f}")
        print(f" > 과거 평균 PER({avg_past_per:.1f}) 기준 현재 적정가: ${current_fair_value:.2f}")
        print(f" > 현재 주가는 적정가 대비 {abs(current_diff):.1f}% {current_status} 상태입니다.")
        print("="*115)
        print(f" ※ 계산 근거: 최근 확정분 합({recent_3_actuals:.2f}) + 야후 분석가 예측({current_q_est:.2f}) = {final_target_eps:.2f}")

    except Exception as e:
        print(f"분석 중 오류 발생: {e}")

if __name__ == "__main__":
    analyze_calendar_final_with_pct()
