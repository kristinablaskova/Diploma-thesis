import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

i=24
stupen=1
old_df = old_dfs[i].dropna(how='any')[:-60]
new_df = new_dfs[i].dropna(how='any')[:-60]
coeffs = np.polyfit(old_df['Minutes'],old_df['Prob'],stupen)
# use line_kws to set line label for legend
ax = sns.regplot(x='Minutes', y='Prob', data=old_df, order=stupen, color='b',
 line_kws={'label':"y={0:.2e}x+{1:.2e}".format(coeffs[0],coeffs[1]) + ' Hradec Kralove'})
coeffs1 = np.polyfit(new_df['Minutes'],new_df['Prob'],stupen)
# use line_kws to set line label for legend
ax = sns.regplot(x='Minutes', y='Prob', data=new_df, order=stupen, color='r',
 line_kws={'label':"y={0:.2e}x+{1:.2e}".format(coeffs1[0],coeffs1[1]) + ' Sleepdata.org'})
# plot legend
ax.legend()
plt.xlim(0,)
plt.ylim(0,1)
plt.ylabel('Probability')
plt.title(str(list_of_pairs[i][0]) + ' to ' + str(list_of_pairs[i][1]))
plt.show()
new_coeffs = new_coeffs.append({'From State': list_of_pairs[i][0], 'To State': list_of_pairs[i][1], 'Coeff': coeffs1},ignore_index=True)
old_coeffs = old_coeffs.append({'From State': list_of_pairs[i][0], 'To State': list_of_pairs[i][1], 'Coeff': coeffs},ignore_index=True)

