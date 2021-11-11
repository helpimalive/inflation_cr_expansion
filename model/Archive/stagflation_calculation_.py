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
    fred    = Fred(r"C:\Users\amcgrady\Documents\GitHub\inflation_cr_expansion\model\APIkey.txt")
        #go to fred cite for id

    data_dict                                                   = {}
    data_dict['Gross Domestic Product']                         ='GDP'
    data_dict['Consumer Price Index']                           ='CPIAUCSL'

        

    data_df         = None

    for k,v in data_dict.items():
        data = fred.get_series_df(v)
        data = pd.DataFrame(data).reset_index()
        data.columns = ['index','realtime_start','realtime_end','date',k]
        clean_data = data.drop(data.columns[[0,1,2]], axis=1)
        if data_df is None:
            data_df=clean_data
        else:
            data_df = pd.merge(data_df, clean_data, how='left', left_on='date', right_on='date')
        
            
    print(data_df.head(20))
    
    GDP = data_df["Gross Domestic Product"]
    GDP = GDP[5:] #make more clear
    GDP = GDP.to_frame()
    GDP['Gross Domestic Product'] = GDP['Gross Domestic Product'].astype(float)
    percent_change_GDP = GDP.pct_change()
    print('Percent change in GDP:')
    print(percent_change_GDP)

    CPI = data_df["Consumer Price Index"]
    CPI = CPI[5:] #make more clear
    CPI = CPI.to_frame()
    CPI['Consumer Price Index'] = CPI['Consumer Price Index'].astype(float)
    percent_change_CPI = CPI.pct_change()
    print('Percent change in CPI:')
    print(percent_change_CPI)

    data_df.index = data_df['date']
    data_df = data_df.rename(columns={'Gross Domestic Product':'GDP'})
    data_df = data_df.rename(columns={'Consumer Price Index':'CPI'})
    data_df=data_df[5:] #make more clear
    data_df['GDP'] = data_df['GDP'].astype(float)
    data_df['GDP'] = data_df['GDP'].pct_change()
    data_df['CPI'] = data_df['CPI'].astype(float)
    data_df['CPI'] = data_df['CPI'].pct_change()
    moving_average_GDP= data_df.GDP.rolling(window=7).mean() #data_df['Moving Average GDP'] =
    moving_average_CPI= data_df.CPI.rolling(window=4).mean()#data_df['Moving Average CPI'] =
    print('GDP smoothed:')
    print(moving_average_GDP)
    print('CPI smoothed:')
    print(moving_average_CPI)

    stagflation_GDP = moving_average_GDP.to_frame()
    stagflation_CPI = moving_average_CPI.to_frame()
    stagflation_index_GDP = []
    stagflation_index_CPI = []
    stagflation_signal = []
    for i in range(len(stagflation_GDP)):
    	if stagflation_GDP.iloc[i,0] < stagflation_GDP.iloc[i-1,0] and stagflation_GDP.iloc[i-2,0]:
    		stagflation_index_GDP.append(i)	
    for i in range(len(stagflation_CPI)):
    	if stagflation_CPI.iloc[i,0] > stagflation_CPI.iloc[i-1,0] and stagflation_CPI.iloc[i-2,0]:
    		stagflation_index_CPI.append(i)	
    stagflation_signal = [i for i in stagflation_index_GDP if i in stagflation_index_CPI]
    print(stagflation_signal)

    confirmed_signal_1 = [x for x in stagflation_signal if x-1 in stagflation_signal]
    confirmed_signal_2 = [x for x in stagflation_signal if x-2 in stagflation_signal]
    confirmed_signal_3 = [x for x in stagflation_signal if x-3 in stagflation_signal]
    confirmed_signal_3 = [x for x in stagflation_signal if x-3 in stagflation_signal]
    confirmed_signal = confirmed_signal_1+confirmed_signal_2+confirmed_signal_3
    confirmed_signal = list(set(confirmed_signal))
    confirmed_signal.sort()
    print(confirmed_signal)

    confirmed_signal_dates=[data_df.iloc[x,0]for x in confirmed_signal]
    print(confirmed_signal_dates)
	

def main():
	_produce_trailing_avgs()

if __name__ == '__main__':
	main()
