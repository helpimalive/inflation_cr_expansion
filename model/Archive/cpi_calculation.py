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


def _produce_trailing_avgs_cpi():
		stag_data = pd.read_excel(r"C:\Users\amcgrady\Documents\Copy of Stagflation for Coding.xlsx", sheet_name='Data', skiprows=0, header=[0])		
		CPI_data = stag_data['CPIAUCSL']
		print(CPI_data)
		fix_CPI_data = CPI_data[1:].dropna()
	
		for x in fix_CPI_data:
			y = ((x-(x-1))/(x-1))
			print(y)

def main():
	_produce_trailing_avgs_cpi()

if __name__ == '__main__':
	main()