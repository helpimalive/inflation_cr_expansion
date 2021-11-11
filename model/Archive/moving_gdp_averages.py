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

def moving_gdp_average():
	API_key = '9e83bfc92bbab5d0086aa294a01fbbae'
	fred = Fred(r"C:\Users\amcgrady\Documents\GitHub\inflation_cr_expansion\model\APIkey.txt")
	 #go to fred cite for id

	data_dict 													= {}
	data_dict['Gross Domestic Product']							='GDP'
	data_dict['Consumer Price Index']							='CPIAUCSL'

	

	data_df 		= None

	for k,v in data_dict.items():
		data = fred.get_series_df(v)
		data = pd.DataFrame(data).reset_index()
		data.columns = ['index','realtime_start','realtime_end','date',k]
		clean_data = data.drop(data.columns[[0,1,2]], axis=1)
		if data_df is None:
	 		data_df=clean_data
		else:
	 		data_df = pd.merge(data_df, clean_data, how='left', left_on='date', right_on='date')
	 		print(data_df)

	GDP = data_df["Gross Domestic Product"]
	GDP = GDP[5:]
	GDP = GDP.to_frame()
	GDP['Gross Domestic Product'] = GDP['Gross Domestic Product'].astype(float)
	percent_change_GDP = GDP.pct_change()
	print(percent_change_GDP)

	CPI = data_df["Consumer Price Index"]
	CPI = CPI[5:]
	CPI = CPI.to_frame()
	CPI['Consumer Price Index'] = CPI['Consumer Price Index'].astype(float)
	percent_change_GDP = CPI.pct_change()
	print(percent_change_GDP)

def main():
	moving_gdp_average()

if __name__ == '__main__':
	main()

