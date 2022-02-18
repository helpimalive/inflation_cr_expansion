from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score,confusion_matrix
import pandas as pd
import numpy as np

# cr = pd.read_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_df_cr.csv')
# flag = pd.read_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_df_flag.csv')
# cr = cr.melt(id_vars='year').dropna()
# flag = flag.melt(id_vars='year').dropna()
# metrics = pd.DataFrame()
# master = pd.merge(flag,cr, left_on=['year','variable'],right_on=['year','variable'])
# master.columns=['year','variable','x','y']


cr = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\national_df_cr.csv')
flag = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\national_df_flag.csv')
metrics = pd.DataFrame()
master = pd.merge(flag,cr, left_on=['date'],right_on=['date'])
master.columns=['year','x','y']
master['year'] = master['year'].apply(lambda x: int(x[0:4]))
# master = master[master['year']>=2007]

for i in range(0,100):
	for i in range(2000,2020,4):
		master_train = master[master['year']<i]
		master_test = master[master['year']>=i]
		X_train = master_train[['x']].astype(int).values
		y_train = np.matrix(master_train['y'].astype(int).values).T
		X_test = master_test[['x']].astype(int).values
		y_test = np.matrix(master_test['y'].astype(int).values).T

		os = SMOTE()
		os_data_X,os_data_y=os.fit_resample(X_train, y_train)
		os_data_X = pd.DataFrame(data=os_data_X)
		os_data_y= pd.DataFrame(data=os_data_y)
		# os_data_y = y_train
		# os_data_X = X_train

		logit_model=sm.Logit(os_data_y,os_data_X)
		result=logit_model.fit()
		var_vals = result.summary2().tables[1]
		cols = var_vals.columns
		var_vals = list(var_vals.values[0])


		pred = np.matrix([int(x>0.5) for x in result.predict(X_test)]).T
		cm = confusion_matrix(y_test, pred)
		tn,fp,fn,tp = cm.ravel()
		print(tn,fp,fn,tp)
		accuracy = (tp+tn)/(tp+fp+fn+tn)
		precision = tp/(tp+fp)
		recall = tp/(tp+fn)
		f_score = 2*(recall*precision)/(recall+precision)
		specificity = tn/(tn+fp)

		var_vals.append(accuracy)
		var_vals.append(precision)
		var_vals.append(recall)
		var_vals.append(f_score)
		var_vals.append(specificity)
		var_vals.insert(0,i)

		metrics = metrics.append(pd.DataFrame([var_vals],columns=['year']+list(cols)+['accuracy','precision','recall','f_score','specificity']))		
	
	os = SMOTE()
	x = master[['x']].astype(int).values
	y = np.matrix(master['y'].astype(int).values).T

	os_data_X,os_data_y=os.fit_resample(x,y)
	os_data_X = pd.DataFrame(data=os_data_X)
	os_data_y= pd.DataFrame(data=os_data_y)
	logit_model=sm.Logit(os_data_y,os_data_X)
	# logit_model=sm.Logit(y,x)
	result=logit_model.fit()
	var_vals = result.summary2().tables[1]
	var_vals = list(var_vals.values[0])

	pred = np.matrix([int(x>=0.5) for x in result.predict(x)]).T
	cm = confusion_matrix(y, pred)
	accuracy = (tp+tn)/(tp+fp+fn+tn)
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f_score = 2*(recall*precision)/(recall+precision)
	specificity = tn/(tn+fp)

	var_vals.append(accuracy)
	var_vals.append(precision)
	var_vals.append(recall)
	var_vals.append(f_score)
	var_vals.append(specificity)
	var_vals.insert(0,'all_time')
	metrics = metrics.append(pd.DataFrame([var_vals],columns=['year']+list(cols)+['accuracy','precision','recall','f_score','specificity']))


metrics.dropna(inplace=True)
metrics = metrics.groupby('year').mean()
print(metrics)
# metrics.to_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_level_accuracy_and_fit.csv')
metrics.to_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\national_level_accuracy_and_fit.csv')