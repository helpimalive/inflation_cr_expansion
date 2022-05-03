from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score,confusion_matrix
import pandas as pd
import numpy as np

# cr = pd.read_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_df_cr.csv')
# flag = pd.read_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_df_flag.csv')
cr = pd.read_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_individual_df_cr.csv')
flag = pd.read_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_individual_df_flag.csv')
cr = cr.melt(id_vars='year').dropna()
flag = flag.melt(id_vars='year').dropna()
metrics = pd.DataFrame()
master = pd.merge(flag,cr, left_on=['year','variable'],right_on=['year','variable'])
master.columns=['year','variable','x','y']

for i in range(2015,2020):
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

	logit_model=sm.Logit(os_data_y,os_data_X)
	result=logit_model.fit()
	var_vals = result.summary2().tables[1]
	cols = var_vals.columns
	var_vals = list(var_vals.values[0])

	pred = np.matrix([int(x>0.5) for x in result.predict(X_test)]).T
	cm = confusion_matrix(y_test, pred)
	tn,fp,fn,tp = cm.ravel()
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

	metrics = metrics.append(pd.DataFrame([var_vals],columns=['year']+list(cols)+['accuracy','precision','recall','F1_score','specificity']))

os = SMOTE()
x = master[['x']].astype(int).values
y = np.matrix(master['y'].astype(int).values).T

os_data_X,os_data_y=os.fit_resample(x,y)
os_data_X = pd.DataFrame(data=os_data_X)
os_data_y= pd.DataFrame(data=os_data_y)

logit_model=sm.Logit(os_data_y,os_data_X)
result=logit_model.fit()
var_vals = result.summary2().tables[1]
cols = var_vals.columns
var_vals = list(var_vals.values[0])
assert False
pred = np.matrix([int(x>0.5) for x in result.predict(master[['x']].astype(int).values)]).T
cm = confusion_matrix(y, pred)
tn,fp,fn,tp = cm.ravel()
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
var_vals.insert(0,'in-sample')
metrics = metrics.append(pd.DataFrame([var_vals],columns=['year']+list(cols)+['accuracy','precision','recall','F1_score','specificity']))

# metrics = metrics.groupby('year').mean().reset_index()
metrics.columns = ['OOS in Year','Coef.','Std.Err.','Z','${P>|z|}$','[0.025','0.975]','accuracy','precision','recall','F1-score','specificity']
metrics.loc[:,~metrics.columns.isin(['${P>|z|}$'])] = metrics.loc[:,~metrics.columns.isin(['${P>|z|}$'])].round(2)
metrics.loc[:,'${P>|z|}$'] = metrics.loc[:,'${P>|z|}$'].apply(lambda x: "%.4f" % x)
 
gof = metrics[['OOS in Year','Coef.','Std.Err.','Z','${P>|z|}$','[0.025','0.975]']]
# gof.to_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_level_fit.csv',index=False,encoding='utf-8-sig')
gof.to_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\individual_msa_level_fit.csv',index=False,encoding='utf-8-sig')
acc = metrics[['OOS in Year','accuracy','precision','recall','F1-score','specificity']]
# acc.to_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\msa_level_accuracy.csv',index=False,encoding='utf-8-sig')
acc.to_csv(r'C:\Users\mlarriva\Documents\GitHub\inflation_cr_expansion\output\individual_msa_level_accuracy.csv',index=False,encoding='utf-8-sig')
