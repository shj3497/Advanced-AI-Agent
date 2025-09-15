# filename: improved_plot_age_vs_pclass.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 다운로드
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
data = pd.read_csv(url)

# 차트 생성
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=data, palette='Set2')
plt.title('Age vs Pclass', fontsize=16)
plt.xlabel('Passenger Class', fontsize=14)
plt.ylabel('Age', fontsize=14)

# 중앙값을 차트에 추가
median_age = data.groupby('pclass')['age'].median()
for pclass in median_age.index:
    plt.text(pclass - 1, median_age[pclass], 
             f'Median: {median_age[pclass]:.1f}', 
             horizontalalignment='center', 
             size='medium', 
             color='black', 
             weight='semibold')

# 차트를 파일로 저장
plt.savefig('age_vs_pclass_improved.png')
plt.savefig('age_vs_pclass_improved.pdf')
plt.close()