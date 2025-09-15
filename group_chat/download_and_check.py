# filename: download_and_check.py
import pandas as pd

# 데이터 다운로드
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
data = pd.read_csv(url)

# 데이터셋의 열 출력
print(data.columns.tolist())