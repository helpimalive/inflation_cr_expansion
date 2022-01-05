import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\matth\\Documents\\GitHub'\
	'\\inflation_cr_expansion\\data\\gdp_cpi_cr_combined.csv')

results = pd.DataFrame(columns = ['mean_per','consec_pers','true_pos','false_negatives'])
for mean_pers in np.arange(1,6):
	for consec_pers in np.arange(1,3):
		df_cpi = df[df['metric']=='cpi']
		df_cpi = df_cpi.pivot(index='year',columns='MSA',values='value')
		df_cpi = df_cpi.pct_change()
		df_mean = df_cpi.rolling(mean_pers,min_periods = mean_pers).mean()
		df_comp = df_cpi>df_mean
		df_consec = df_comp.rolling(consec_pers).sum()==(consec_pers)
		df_cpi_flag = df_consec[~df_mean.isna()] 

		df_gdp = df[df['metric']=='gdp']
		df_gdp = df_gdp.pivot(index='year',columns='MSA',values='value')
		df_gdp = df_gdp.pct_change()
		df_mean = df_gdp.rolling(mean_pers,min_periods = mean_pers).mean()
		df_comp = df_gdp<df_mean
		df_consec = df_comp.rolling(consec_pers).sum()==(consec_pers)
		df_gdp_flag = df_consec[~df_mean.isna()] 
		df_flag = (df_gdp_flag==1)&(df_cpi_flag==1)

		df_cr = df[df['metric']=='cap_rate']
		df_cr = df_cr.pivot(index='year',columns='MSA',values='value')
		df_cr = df_cr.diff()
		df_cr = df_cr>0
		
		# We'll consider a cap rate expansion one that happens in either 
		# of the next two years
		df_cr = df_cr.rolling(2).sum()
		df_cr = df_cr.shift(-2)
		df_cr = df_cr>0

		mutual_dates = set(df_flag.index).intersection(df_cr.index)
		df_flag = df_flag[df_flag.index.isin(mutual_dates)]
		positive_accuracy = df_flag==df_cr
		positive_accuracy = positive_accuracy[df_flag==True]
		true_positives = positive_accuracy.sum().sum()/positive_accuracy.count().sum()
		false_positives = 1-true_positives

		df_cr = df[df['metric']=='cap_rate']
		df_cr = df_cr.pivot(index='year',columns='MSA',values='value')
		df_cr = df_cr.diff()
		df_cr = df_cr>0
		df_cr = df_cr.shift(-1)
		df_cr = df_cr.iloc[0:df_cr.shape[0]-1]
		df_flag = df_flag.iloc[0:df_flag.shape[0]-1]
		negative_accuracy= df_flag==df_cr
		negative_accuracy= negative_accuracy[df_flag==False]
		miss = negative_accuracy==0
		miss = miss.sum().sum()
		expansions = negative_accuracy!=1
		expansions = expansions.sum().sum()
		false_negatives = miss/expansions
		# negative_accuracy = 

		# true_negatives = accuracy.count().sum()/

		trial = pd.DataFrame([[mean_pers,
			consec_pers,
			true_positives,
			false_negatives,
			]],columns = results.columns)
		results = pd.concat([results,trial])

print(results[results['true_pos']==results.true_pos.max()])

