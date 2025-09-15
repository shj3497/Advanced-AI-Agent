# filename: final_improved_plot_age_vs_pclass.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 다운로드
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
data = pd.read_csv(url)

# 차트 생성
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', hue='pclass', data=data, palette='Set2', dodge=False)
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

# 배경 및 격자 설정
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.gca().set_facecolor('whitesmoke')

# 차트를 파일로 저장
plt.savefig('final_age_vs_pclass.png')
plt.savefig('final_age_vs_pclass.pdf')
plt.close()