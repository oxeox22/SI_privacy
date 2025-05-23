import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

# ---------- 유틸리티 함수 ----------

def select_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="CSV 파일 선택", filetypes=[("CSV files", "*.csv")])

def categorize(series, bins):
    edges = np.linspace(series.min(), series.max(), bins + 1)
    return pd.cut(series, bins=edges, labels=False, include_lowest=True), edges

def clip_and_categorize(series, bins, bottom=None, top=None):
    series = series.clip(lower=bottom, upper=top)
    return categorize(series, bins)

def midpoint_from_bins(series, bins, bottom=None, top=None):
    if bottom is not None:
        series = series.clip(lower=bottom)
    if top is not None:
        series = series.clip(upper=top)
    edges = np.linspace(series.min(), series.max(), bins + 1)
    mids = [(edges[i] + edges[i+1]) / 2 for i in range(bins)]
    categories = pd.cut(series, bins=edges, labels=False, include_lowest=True)
    return categories.map(lambda x: mids[int(x)] if pd.notnull(x) and 0 <= x < len(mids) else np.nan)

def is_k_anonymous(df, k):
    return (df.groupby(df.columns.tolist()).size() >= k).all()

def compute_sse(original_df, approx_df):
    return ((original_df - approx_df) ** 2).sum().sum()

# ---------- 실행 파트 ----------

input_file = select_file()
if not input_file:
    raise Exception("파일이 선택되지 않았습니다.")
print(f"\n선택된 파일: {input_file}")

# 사용자 입력
k_val = int(input("K-익명성 기준(K)을 입력하세요 (예: 3): "))
quasi_columns = input("준식별자 컬럼명들을 쉼표로 입력하세요 (예: 나이,소득,가족 수): ").strip().split(",")
sensitive_columns = input("민감정보 컬럼명들을 쉼표로 입력하세요 (예: 신용 대출 여부): ").strip().split(",")

# 데이터 불러오기
try:
    df = pd.read_csv(input_file, encoding='cp949')
except UnicodeDecodeError:
    df = pd.read_csv(input_file, encoding='utf-8-sig')

df_selected = df[quasi_columns + sensitive_columns].copy()

# 컬럼 자동 리네이밍
quasi_renamed = [f'q{i}' for i in range(len(quasi_columns))]
rename_map = dict(zip(quasi_columns, quasi_renamed))
df_selected.rename(columns=rename_map, inplace=True)
df_selected.columns = quasi_renamed + sensitive_columns
original = df_selected[quasi_renamed]

# 수치형 컬럼 필터링
numeric_quasi = [col for col in quasi_renamed if pd.api.types.is_numeric_dtype(df_selected[col])]
if not numeric_quasi:
    raise Exception("수치형 준식별자 컬럼이 없습니다.")

# 전략 및 파라미터
strategies = ['bin_only', 'clip_and_bin']
bin_options = [3, 4, 5, 6]
bottom_ratios = [None, 0.05]
top_ratios = [None, 0.95]

# 각 컬럼에 대해 유효한 bin 수만 필터링 (고유값 수보다 적은 것만)
column_bin_options = {}
for col in numeric_quasi:
    unique_vals = df_selected[col].nunique()
    column_bin_options[col] = [b for b in bin_options if b < unique_vals]
    if not column_bin_options[col]:
        print(f"'{col}' 컬럼은 유효한 bin 수 없음 (고유값: {unique_vals}) → 생략")
        numeric_quasi.remove(col)

# 조합 생성
strategy_combinations = list(itertools.product(*[strategies]*len(numeric_quasi)))
bin_combinations = list(itertools.product(*[column_bin_options[col] for col in numeric_quasi]))
total_combinations = len(strategy_combinations) * len(bin_combinations) * len(bottom_ratios) * len(top_ratios)

results = []
progress_bar = tqdm(total=total_combinations, desc="🔍 탐색 중...", ncols=100)

for strat_combo in strategy_combinations:
    for bin_combo in bin_combinations:
        for bottom_ratio in bottom_ratios:
            for top_ratio in top_ratios:
                try:
                    df_temp = df_selected.copy()
                    applied_strategy = {}

                    for i, col in enumerate(numeric_quasi):
                        strat = strat_combo[i]
                        bins = bin_combo[i]
                        bottom = df_selected[col].quantile(bottom_ratio) if bottom_ratio is not None else None
                        top = df_selected[col].quantile(top_ratio) if top_ratio is not None else None

                        if strat == 'clip_and_bin':
                            df_temp[col], _ = clip_and_categorize(df_temp[col], bins, bottom, top)
                        else:
                            df_temp[col], _ = categorize(df_temp[col], bins)

                        applied_strategy[col] = {
                            'strategy': strat,
                            'bins': bins,
                            'bottom': bottom,
                            'top': top
                        }

                    if is_k_anonymous(df_temp[numeric_quasi], k_val):
                        approx = pd.DataFrame()
                        for col in numeric_quasi:
                            info = applied_strategy[col]
                            approx[col] = midpoint_from_bins(df_selected[col], info['bins'], info['bottom'], info['top'])
                        sse = compute_sse(original[numeric_quasi], approx[numeric_quasi])
                        record = {
                            f'{col}_strategy': applied_strategy[col]['strategy'] for col in numeric_quasi
                        }
                        record.update({
                            f'{col}_bins': applied_strategy[col]['bins'] for col in numeric_quasi
                        })
                        record.update({
                            f'{col}_bottom': applied_strategy[col]['bottom'] for col in numeric_quasi
                        })
                        record.update({
                            f'{col}_top': applied_strategy[col]['top'] for col in numeric_quasi
                        })
                        record['sse'] = sse
                        results.append(record)
                except Exception as e:
                    print(f"오류 발생: {e}")
                finally:
                    progress_bar.update(1)

progress_bar.close()

# 결과 정리
results_df = pd.DataFrame(results)
if results_df.empty:
    print(f"\nK={k_val} 만족하는 조합이 없습니다.")
    exit()

means = original[numeric_quasi].mean()
approx_mean = pd.DataFrame(np.tile(means.values, (len(original), 1)), columns=numeric_quasi)
max_sse = compute_sse(original[numeric_quasi], approx_mean)
results_df['utility_score'] = 1 - (results_df['sse'] / max_sse)
results_df = results_df.sort_values(by='sse')
best = results_df.iloc[0]

print("\n🎯 최적 조합:")
print(best)

print(f"\nTop 5 조합:")
print(results_df.head(5))

# 최종 데이터 생성
df_anonymized = pd.DataFrame()
for col in numeric_quasi:
    bins = int(best[f'{col}_bins'])
    bottom = best[f'{col}_bottom']
    top = best[f'{col}_top']
    df_anonymized[col] = midpoint_from_bins(df_selected[col], bins, bottom=bottom, top=top)

# 민감정보 추가 & 컬럼명 복원
df_anonymized[sensitive_columns] = df_selected[sensitive_columns]
reverse_map = {v: k for k, v in rename_map.items()}
df_anonymized.rename(columns=reverse_map, inplace=True)

# 저장
output_path = input_file.replace(".csv", "_anonymized.csv")
df_anonymized.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n비식별화된 데이터가 저장되었습니다: {output_path}")
