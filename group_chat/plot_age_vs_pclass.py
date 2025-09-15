# filename: plot_age_vs_pclass.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 다운로드
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
data = pd.read_csv(url)

# 차트 생성
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=data)
plt.title('Age vs Pclass')
plt.xlabel('Passenger Class')
plt.ylabel('Age')

# 차트를 파일로 저장
plt.savefig('age_vs_pclass.png')
plt.close()