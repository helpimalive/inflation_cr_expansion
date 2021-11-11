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

def cap_rate_calculation():
 #with _exit_on_error("file_path", Exception):
 	cr_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","green_street_cr_data.xlsx")
 #with _exit_on_error("read_in", Exception):
 	cr_data = pd.read_excel(cr_path)
 	output_col = cr_data.iloc[:,0].str.extract(r'(\d\d\d\d-\d\d-\d\d)')
 	output_col = output_col.dropna()
 	start_row = output_col.index.values.min()
 	end_row = output_col.index.values.max()
 	title_row = start_row - 1
 	number_of_columns = len(cr_data.loc[title_row:,])
 	columns_names = cr_data.iloc[title_row, : number_of_columns]
 	clean_cr_data = cr_data.loc[start_row:end_row]
 	clean_cr_data.columns = columns_names.values
 	cr_data = clean_cr_data['Apartment']
 	print(cr_data)

def main():
	cap_rate_calculation()

if __name__ == '__main__':
	main()