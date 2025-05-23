# SI_privacy
# K-Anonymity SSE Optimizer
> 다양한 범주화(bin) 및 클리핑(clip) 전략을 실험해 K-익명성을 만족하면서 원본 대비 SSE(제곱합 오차)를 최소화하는 비식별화 파이프라인

## 🚀 주요 기능
- **CSV 선택 인터페이스**  
  - `tkinter` 파일 다이얼로그로 처리할 CSV 파일 지정  
- **K-익명성 검증**  
  - 사용자가 입력한 준식별자(Quasi-Identifiers)에 대해 K-익명성 만족 여부 확인  
- **범주화 전략 비교**  
  - `bin_only` vs `clip_and_bin` 전략을 각 컬럼별로 조합하여 실험  
- **파라미터 탐색**  
  - bin 개수, 하위/상위 클리핑 비율(bottom/top quantile)을 조합한 모든 경우의 수 평가  
- **SSE 계산 & 유틸리티 스코어**  
  - 원본 데이터와 근사(중간값 기반) 데이터 간 SSE 계산  
  - 최대 SSE 대비 상대 점수(utility_score) 산출  
- **최적 조합 자동 적용**  
  - SSE가 최소인 전략 조합으로 최종 비식별화 데이터 생성 및 CSV로 저장  

## 📦 기술 스택
- Python 3.7+  
