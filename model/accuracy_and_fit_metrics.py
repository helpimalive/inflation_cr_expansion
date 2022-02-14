from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score,confusion_matrix
import pandas as pd
import numpy as np

cpi = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\msa_df_cpi_flag.csv')
gdp = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\msa_df_gdp_flag.csv')
cr = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\msa_df_cr.csv')
flag = pd.read_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\msa_df_flag.csv')

cr = cr.melt(id_vars='year').dropna()
flag = flag.melt(id_vars='year').dropna()
gdp = gdp.dropna().melt(id_vars='year').dropna()
cpi = cpi.melt(id_vars='year').dropna()

metrics = pd.DataFrame()

for i in range(2010,2020):
	master = pd.merge(flag,cr, left_on=['year','variable'],right_on=['year','variable'])
	master.columns=['year','variable','x','y']
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


	pred = np.matrix([int(x>=0.5) for x in result.predict(X_test,linear=True)]).T
	acs = accuracy_score(y_test,pred)
	bal = balanced_accuracy_score(y_test,pred)
	cm = confusion_matrix(y_test, pred)
	tn,fp,fn,tp = cm.ravel()
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)

	var_vals.append(acs)
	var_vals.append(bal)
	var_vals.append(precision)
	var_vals.append(recall)
	var_vals.insert(0,i)

	metrics = metrics.append(pd.DataFrame([var_vals],columns=['year']+list(cols)+['accuracy']+['balanced_accuracy']+['precision']+['recall']))

print(metrics.mean())

master = pd.merge(flag,cr, left_on=['year','variable'],right_on=['year','variable'])
master.columns=['year','variable','x','y']
os = SMOTE()
os_data_X,os_data_y=os.fit_resample(master[['x']].astype(int),master[['y']].astype(int))
os_data_X = pd.DataFrame(data=os_data_X)
os_data_y= pd.DataFrame(data=os_data_y)
logit_model=sm.Logit(os_data_y,os_data_X)
result=logit_model.fit()

var_vals = result.summary2().tables[1]
var_vals = list(var_vals.values[0])
pred = np.matrix([int(x>=0.5) for x in result.predict(master[['x']].astype(int),linear=True)]).T
acs = accuracy_score(master[['y']].astype(int),pred)
bal = balanced_accuracy_score(master[['y']].astype(int),pred)
tn, fp, fn, tp = confusion_matrix(master[['y']].astype(int), pred).ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
var_vals.append(acs)
var_vals.append(bal)
var_vals.append(precision)
var_vals.append(recall)
var_vals.insert(0,'all_time')

metrics = metrics.append(pd.DataFrame([var_vals],columns=['year']+list(cols)+['accuracy']+['balanced_accuracy']+['precision']+['recall']))
metrics.to_csv(r'C:\Users\matth\Documents\GitHub\inflation_cr_expansion\output\msa_level_accuracy_and_fit.csv')