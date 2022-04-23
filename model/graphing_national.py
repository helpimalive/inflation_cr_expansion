import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

cr = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\data\National_CR_GDP_CPI.csv')
cr.date = pd.to_datetime(cr.date)

flag = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\national_df_flag.csv')
flag.date = pd.to_datetime(flag.date)
flag.date = flag.date + pd.offsets.DateOffset(years=1)
flag.columns = ['date','forecast']

cr = cr[cr['date']>=flag.date.min()]

df_outperformance = pd.DataFrame()
strat_cr = 0
true_positives = 0
for r in range(0,len(flag)-4):
	cr['difference'] = cr.CR.diff(4)
	x_0 = str(cr.iloc[r]['date'])
	x_1 = str(cr.iloc[r+4]['date'])

	y_0 = cr.iloc[r]['CR']
	y_1 = cr.iloc[r+4]['CR']

	sign = cr.iloc[r+4]['difference']
	color = 'green' if sign<0 else 'red'
	x,y = zip([x_0,y_0],[x_1,y_1])
	ax[row,col].plot(x,y,color = color,linewidth=5)

	# True Positive
	if flag.iloc[r]['forecast'] and sign>0:
		ax[row,col].axvspan(x_0,x_1,facecolor='green', alpha=0.5)
		strat_cr+=0
		true_positives+=1
	# True Negative
	elif not flag.iloc[r]['forecast'] and sign<0:
		ax[row,col].axvspan(x_0,x_1,facecolor='green', alpha=0.5)
		strat_cr+=sign
	# False Positive
	elif flag.iloc[r]['forecast'] and sign<0:
	 	ax[row,col].axvspan(x_0,x_1,facecolor='red', alpha=0.5)
	 	strat_cr+=0
	# False Negative
	elif not flag.iloc[r]['forecast'] and sign>0:
		ax[row,col].axvspan(x_0,x_1,facecolor='red', alpha=0.5)
		strat_cr+=sign

outperformance = strat_cr/one_msa.difference.sum()-1
outperformance = "{:.0%}".format(outperformance)
expansions = [1 for x in one_msa.difference if x>0]
expansions = np.sum(expansions)
if i>0:
	df_outperformance = pd.concat([df_outperformance,pd.DataFrame([[msa,outperformance]])],axis=0,ignore_index=True)
else:
	df_outperformance = pd.DataFrame([[msa,outperformance]])
ax[row,col].set_title(f"{msa} \n Strategy Out(under)performance vs Buy-and-Hold = {outperformance} \n CR Expansions Captured = {true_positives} out of {expansions}",fontsize=7)
vals = ax[row,col].get_yticks()
ax[row,col].set_yticklabels(['{:,.2%}'.format(x) for x in vals],fontsize=7)

ax[row,col].set_xticks(ax[row,col].get_xticks()[::2])
ax[row,col].set_xticklabels(one_msa['year'].values[::2])
ax[row,col].set_xticklabels(ax[row,col].get_xticklabels(),fontsize=7)

ax[row,col].tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False         # ticks along the top edge are off
    ) 
i+=1
legend_elements = [Line2D([0], [0], color='green', lw=2, label='CR decrease'),
			Line2D([0], [0], color='red', lw=2, label='CR increase'),
           Patch(facecolor='green', 
                 label='correct forcast'),
           Patch(facecolor='red', 
                 label='incorrect forecast')]

fig.legend(handles=legend_elements,loc='center',bbox_to_anchor=(.475,0.90))
# df_outperformance.columns=['MSA','Outperformance vs Buy and Hold']
# df_outperformance.to_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\outperformance.csv',index=False)

