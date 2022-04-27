import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

cr = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\data\National_CR_GDP_CPI.csv')
cr.date = pd.to_datetime(cr.date)
cr['fwd_change'] = cr['CR'].diff(4)
cr['next_change'] = cr['CR'].diff(1)

flag = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\national_df_flag.csv')
flag.date = pd.to_datetime(flag.date)
flag.date = flag.date + pd.offsets.DateOffset(years=1)
flag.columns = ['date','forecast']

cr = cr[cr['date']>=flag.date.min()]
comb = pd.merge(cr,flag)
comb=comb[['year','date','CR','fwd_change','forecast','next_change']]

df_outperformance = pd.DataFrame()
strat_cr = 0
true_positives = 0
fig,ax = plt.subplots()

for r in range(0,len(comb)-1):
	x_0 = str(comb.iloc[r]['date'])
	x_1 = str(comb.iloc[r+1]['date'])

	y_0 = comb.iloc[r]['CR']
	y_1 = comb.iloc[r+1]['CR']

	sign = comb.iloc[r]['fwd_change']

	color = 'green' if (y_1-y_0)<0 else 'red'
	x,y = zip([x_0,y_0],[x_1,y_1])
	ax.plot(x,y,color = color,linewidth=5)

	# True Positive
	if comb.iloc[r]['forecast'] and sign>0:
		ax.axvspan(x_0,x_1,facecolor='green', alpha=0.5)
		true_positives+=1
	# True Negative
	elif not comb.iloc[r]['forecast'] and sign<0:
		ax.axvspan(x_0,x_1,facecolor='green', alpha=0.5)
	# False Positive
	elif comb.iloc[r]['forecast'] and sign<0:
	 	ax.axvspan(x_0,x_1,facecolor='red', alpha=0.5)
	# False Negative
	elif not comb.iloc[r]['forecast'] and sign>0:
		ax.axvspan(x_0,x_1,facecolor='red', alpha=0.5)

strat_cumulative_cr = comb[comb['forecast']==False]['fwd_change'].sum()
cumulative_cr = comb.fwd_change.sum() 
outperformance = strat_cumulative_cr/cumulative_cr-1
outperformance = "{:.0%}".format(outperformance)
expansions = comb[comb['fwd_change']>0]['fwd_change'].count()
ax.set_title(f"Strategy Outperformance vs Buy-and-Hold = {outperformance} \n Accuracy = 70%  {true_positives} CR Expansions Captured out of {expansions}",fontsize=9)
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals],fontsize=7)

ax.set_xticks(ax.get_xticks()[::8])
ax.set_xticklabels(comb['year'].values[::8])
ax.set_xticklabels(ax.get_xticklabels(),fontsize=7)

# ax.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,
#     left=False         # ticks along the top edge are off
#     ) 
# i+=1
legend_elements = [Line2D([0], [0], color='green', lw=2, label='CR decrease'),
			Line2D([0], [0], color='red', lw=2, label='CR increase'),
           Patch(facecolor='green', 
                 label='correct forcast'),
           Patch(facecolor='red', 
                 label='incorrect forecast')]


plt.show()
plt.figure(figsize=(3.841, 7.195),dpi=200)
fig.savefig(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\national_performance.jpg')
