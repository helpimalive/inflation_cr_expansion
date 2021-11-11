import argparse
import collections
import contextlib
import csv
import datetime
import enum
import operator
import os
import pickle
import sys
import re
import pathlib
import shutil
import time
import numpy as np
import pandas as pd
from datetime import datetime
from full_fred.fred import Fred
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import requests
import json
from blsconnect import RequestBLS, bls_search


def _produce_trailing_avgs():
	API_key = '9e83bfc92bbab5d0086aa294a01fbbae'
	fred 	= Fred(r"C:\Users\amcgrady\Documents\GitHub\inflation_cr_expansion\model\APIkey.txt")
	#go to fred cite for id
	data_dict_fred={}

	data_dict_fred['GDP']                                                   ={}
	#data_dict_fred['GDP']['US Total RGDP']							     	='GDPC1'                 
	data_dict_fred['GDP']['Philadelphia Camden Wilmington PA NJ DE MD']      ='RGMP37980'  
	data_dict_fred['GDP']['Dallas Fort Worth Arlington TX']                 ='RGMP19100'
	data_dict_fred['GDP']['Houston The Woodlands Sugar Land TX']            ='RGMP26420'
	data_dict_fred['GDP']['Phoenix Mesa Scottsdale AZ']                     ='RGMP38060'
	data_dict_fred['GDP']['Denver Aurora Lakewood CO']                      ='RGMP19740'
	data_dict_fred['GDP']['Seattle Tacoma Bellevue WA']                     ='RGMP42660'
	data_dict_fred['GDP']['New York Newark Jersey City NY NJ PA']           ='RGMP35620'
	data_dict_fred['GDP']['San Francisco Oakland Hayward CA']               ='RGMP41860'
	data_dict_fred['GDP']['Chicago Naperville Elgin IL IN WI']               ='RGMP16980'
	data_dict_fred['GDP']['Tampa St.Petersburg Clearwater FL']              ='RGMP45300'
	data_dict_fred['GDP']['Atlanta Sandy Spirngs Roswell GA']               ='RGMP12060'
	data_dict_fred['GDP']['Miami For Lauderdale West Palm Beach FL']        ='RGMP33100'
	data_dict_fred['GDP']['Boston Cambridge Newton MA NH']                  ='RGMP14460'
	data_dict_fred['GDP']['San Diego Carlsbad CA']                          ='RGMP41740'
	data_dict_fred['GDP']['Detroit Warren Dearborn MI']                     ='RGMP19820'
	data_dict_fred['GDP']['St.Louis Mo IL']                                 ='RGMP41180'
	data_dict_fred['GDP']['Minneapolis St.Paul Bloomington MN WI']          ='RGMP33460'
	data_dict_fred['GDP']['Washington Arlington Alexandria DC VA MD WV']    ='RGMP47900'
	data_dict_fred['GDP']['Baltimore Columbia Towson MD']                   ='RGMP12580'
	data_dict_fred['GDP']['Los Angeles Long Beach Anaheim CA']              ='RGMP31080'
	# print(data_dict_fred)

	data_dict_fred['CPI']                                                      ={}
	data_dict_fred['CPI']['US Total CPI']						     			='CPIAUCSL'
	data_dict_fred['CPI']['Philadelphia Camden Wilmington PA NJ DE MD']       ='CUURA102SA0'
	data_dict_fred['CPI']['Dallas Fort Worth Arlington TX']                   ='CUURA316SA0'
	data_dict_fred['CPI']['Houston The Woodlands Sugar Land TX']              ='CUURA318SA0'
	data_dict_fred['CPI']['Phoenix Mesa Scottsdale AZ']                       ='CUUSA429SA0'
	data_dict_fred['CPI']['Denver Aurora Lakewood CO']                        ='CUUSA433SEHA'            
	data_dict_fred['CPI']['Seattle Tacoma Bellevue WA']                       ='CUURA423SEHA'
	data_dict_fred['CPI']['New York Newark Jersey City NY NJ PA']             ='CUURA101SA0'
	data_dict_fred['CPI']['San Francisco Oakland Hayward CA']                 ='CUURA422SA0'
	data_dict_fred['CPI']['Chicago Naperville Elgin IL IN WI']                ='CUURA207SA0'
	data_dict_fred['CPI']['Tampa St.Petersburg Clearwater FL']                ='CUUSA321SA0'
	data_dict_fred['CPI']['Atlanta Sandy Spirngs Roswell GA']                 ='CUURA319SA0'
	data_dict_fred['CPI']['Miami For Lauderdale West Palm Beach FL']          ='CUURA320SA0'
	data_dict_fred['CPI']['Boston Cambridge Newton MA NH']                    ='CUUSA103SA0'
	data_dict_fred['CPI']['San Diego Carlsbad CA']                            ='CUUSA424SA0'
	data_dict_fred['CPI']['Detroit Warren Dearborn MI']                       ='CUURA208SA0'
	data_dict_fred['CPI']['St.Louis Mo IL']                                   ='CUUSA209SA0'
	data_dict_fred['CPI']['Minneapolis St.Paul Bloomington MN WI']            ='CUUSA211SA0'
	

	#master_dict = {MSA_name = {msa_key_bls =XXXX, msa_key_fred = XXXX}}

	# master_dict={}
	# master_dict={Philadelphia_Camden_Wilmington_PA_NJ_DE_MD = {GDP_fred='RGMP37980',CPI='CUURA102SA0'},{Dallas_Fort_Worth_Arlington_TX={'RGMP19100','CUURA316SA0'}},Houston_The_Woodlands_Sugar_Land_Tx={'RGMP26420','CUURA318SA0'},
	# Phoenix_Mesa_Scottsdale_AZ={'RGMP38060', 'CUUSA429SA0'}, Denver_Aurora_Lakewood_CO={'RGMP19740','CUURA423SEHA'},New_York_Newark_Jersey_City_NY_NJ_PA={'RGMP35620','CUURA101SA0'},
	# Seattle_Tacoma_Bellevue_WA={'RGMP42660','CUURA423SEHA'}, San_Francisco_Oakland_Hayward_CA={'RGMP41860','CUURA422SA0'}, Chicago_Naperville_Elgin_IL_IN_WI={'RGMP16980','CUURA207SA0'},
	# Tampa_St.Petersburg_Clearwater_FL={'RGMP45300','CUUSA321SA0'}, Atlanta_Sandy_Spirngs_Roswell_GA={'RGMP12060','CUURA319SA0'}}

	# print(master_dict.items())
	# for k,v in master_dict.items():
	# 	data = fred.get_series_df(v)
	# 	data = pd.DataFrame(data).reset_index()


	API_key_bls ="4bcde33801484b78a194e85816e17cc9"
	bls = RequestBLS(key=API_key_bls)
	months = ['M01']
	
	# Washington = bls.series('CUURS35ASA0', start_year = 1946, end_year = 2021)
	# Washington.drop(columns=['periodName'], inplace=True)
	# Washington = Washington.rename(columns={'CUURS35ASA0':'Washington Arlington Alexandria DC VA MD WV'})
	# Washington = Washington.rename(columns={'year':'date'})
	# Washington['marker'] = Washington.period.isin(quarter_months).astype(str)
	# Washington.columns = ['Date',  'period_x', 'Washington Arlington Alexandria DC VA MD WV', 'marker']
	# Washington.drop(Washington.loc[Washington['marker']=='False'].index, inplace=True)
	# Washington= Washington.drop(Washington.columns[[1,3]], axis=1)
	# #print(Washington)

	# LA = bls.series('CUURS49ASA0', start_year = 1946, end_year = 2021)
	# LA.drop(columns=['periodName'], inplace=True)
	# LA = LA.rename(columns={'CUURS49ASA0':'Los Angeles Long Beach Anaheim CA'})
	# LA = LA.rename(columns={'year' : 'date'})
	# LA['marker'] = LA.period.isin(months).astype(str)
	# LA.columns = ['Date',  'period_y', 'Los Angeles Long Beach Anaheim CA', 'marker']
	# LA.drop(LA.loc[LA['marker']=='False'].index, inplace=True)
	# LA= LA.drop(LA.columns[[1,3]], axis=1)
	# print(LA)


	# Baltimore = bls.series('CUURS35ESA0', start_year = 1946, end_year = 2021)
	# Baltimore.drop(columns=['periodName'], inplace=True)
	# Baltimore = Baltimore.rename(columns={'CUURS35ESA0':'Baltimore Columbia Towson MD'})
	# Baltimore = Baltimore.rename(columns={'year' : 'date'})
	# Baltimore['marker'] = Baltimore.period.isin(months).astype(str)
	# Baltimore.columns = ['Date',  'period_z', 'Baltimore Columbia Towson MD', 'marker']
	# #Baltimore_clean = Baltimore.drop(Baltimore.columns[[1,3]], axis=1)
	# Baltimore.drop(Baltimore.loc[Baltimore['marker']=='False'].index, inplace=True)
	# Baltimore = Baltimore.drop(Baltimore.columns[[1,3]], axis=1)
	# print(Baltimore.head(14))

	# missing_cpi = Washington.merge(Washington,LA, left_on='Date', right_on='Date')
	# #missing_cpi = missing_cpi.merge(Baltimore, on='Date')
	# print(missing_cpi.head(14))
	
	#missing_cpi.columns = ['Washington DC CPI', 'Los Angeles CPI']
	# #print(missing_cpi)
	data_df 		= None

	for k,v in data_dict_fred.items():
		use_dict = v
		for i, j in use_dict.items():
			data= fred.get_series_df(j)
			data = pd.DataFrame(data).reset_index()
			clean_data = data.drop(data.columns[[0,1,2]], axis=1)
			clean_data.columns = ['Date', i]
			#print(clean_data)
			if data_df is None:
				data_df=clean_data
			else:
				data_df = data_df.merge(clean_data, on = 'Date')
		print(data_df)

			
	# data_df = pd.concat([data_df, missing_cpi], axis=1, ignore_index=False)
	# data_df = pd.concat([data_df, Baltimore], axis=1, ignore_index=False)
	#data_df = data_df.merge(missing_cpi, on='Date')
	#print(data_df)


	philly1 = data_df[1].to_frame()
	philly_GDP_chng = philly1.astype(float)
	philly_GDP_chng = philly_GDP_chng.pct_change()

	dallas1 = data_df['Dallas Fort Worth Arlington TX'].to_frame()
	dallas_GDP_chng = dallas1.astype(float)
	dallas_GDP_chng = dallas_GDP_chng.pct_change()

	data_df_msa_gdp = data_df.drop(data_df.columns[[0]], axis=1)
	data_df_msa_gdp = data_df_msa_gdp.iloc[0:,:39]
	#print(data_df_msa_gdp)
	

	# #data_df_msa_gdp.index = data_df_msa_gdp['Date']
	# for (columnName, columnData) in data_df_msa_gdp.iteritems():
	# 	columnData = columnData
	# 	print('Percent Change :', columnName)
	# 	print(columnData.pct_change())

def main():
	_produce_trailing_avgs()

if __name__ == '__main__':
	main()
