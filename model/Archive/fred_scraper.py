def pull_data():

	API_key = '9e83bfc92bbab5d0086aa294a01fbbae'
	fred 	= Fred(api_key=API_key)

	data_dict 													= {}
	data_dict['Total Nonfarm Payrolls']							='PAYEMS'
	data_dict['Hourly Earninigs']								='CES0500000003'
	data_dict['Median usual weekly real earnings']				='LES1252881600Q'
	data_dict['Consumer Price Index']							='CPIAUCSL'
	data_dict['Import Price Index']								='IR'
	data_dict['Total Nonfarm Payrolls']							='PAYEMS'
	data_dict['Producer Price Index']							='PPIACO'
	data_dict['Unemployment Rate']								='UNRATE'
	data_dict['Home Price Index']								='CSUSHPISA'
	data_dict['Delinquencies']									='DALLSRESFRMACBEP'
	data_dict['Banks Tightening Standards for Subprime']		='DRTSSP'
	data_dict['Exst Home Sales Supply Months']					='HOSSUPUSM673N'
	data_dict['Housing Starts']									='HOUST'
	data_dict['Housing Starts 5 or More']						='HOUST5F'
	data_dict['Housing Starts Midwest']							='HOUSTMW'
	data_dict['Housing Starts Northeast']						='HOUSTNE'
	data_dict['Housing Starts South']							='HOUSTS'
	data_dict['Housing Starts West']							='HOUSTW'
	data_dict['Monthly Supply of Houses']						='MSACSR'
	data_dict['New One Family Home Sales']						='HSN1F'
	data_dict['Existing Home Sales'] 							='EXHOSLUSM495S'
	data_dict['Reporting Stronger Demand for Gov Mtg Loans']	='SUBLPDHMDGNQ'
	data_dict['Tightening Standards for Gov Mtg Loans']			='SUBLPDHMSGNQ'
	data_dict['Tightening Standards for Qualified Jumbo']		='SUBLPDHMSJNQ'
	data_dict['Tightening Standards for NonQualified Jumbo']	='SUBLPDHMSKNQ'
	data_dict['Tightening Standards for Qualified Non-Jumbo']	='SUBLPDHMSQNQ'
	data_dict['Real Disposable Personal Income']				='A229RX0'
	data_dict['30 Year Mortgage Rate']							='MORTGAGE30US'
	data_dict['HH DS % Disposable Income']						='TDSP'
	data_dict['Open Job Listings']								='JTSJOL'
	data_dict['CPI Rent of Shelter']							='CUSR0000SAS2RS'
	data_dict['Homeownership Rate']								='RHORUSQ156N'
	data_dict['Renter Occupied Housing']						='ERNTOCCUSQ176N'
	data_dict['Vacant Housing Units']							='ERNTSLDUSQ176N'
	data_dict['National Home Price Index']						='CSUSHPINSA'	
	data_dict['Personal Saving Rate']							='PSAVERT'	
	data_dict['Permits 5-units or more']						='PERMIT5'
	data_dict['Permits 1-unit']									='PERMIT1'
	data_dict['High-Propensity Business Appl']					='HPBUSAPPSAUS'
	data_dict['Business Appl']									='BUSAPPSAUS'

	data_df 		= pd.DataFrame()
	base 			= pd.to_datetime('1920-01-01')
	date_list 		= [base + relativedelta(months=x) for x in range(0, 1500)]
	data_df['date'] = date_list
	quarter_sets	= [	'Median usual weekly real earnings',
						'Reporting Stronger Demand for Gov Mtg Loans',
						'Tightening Standards for Gov Mtg Loans',
						'Tightening Standards for Qualified Jumbo',
						'Tightening Standards for NonQualified Jumbo',
						'Tightening Standards for Qualified Non-Jumbo',
						'Delinquencies',
						'Homeownership Rate',
						'Renter Occupied Housing',
						'Vacant Housing Units',
						'CPI Rent of Shelter']
	month_into_quarter_sets = ['30 Year Mortgage Rate']

	for k,v in data_dict.items():
		data = fred.get_series(v)
		data = pd.DataFrame(data).reset_index()
		data.columns = ['date',k]
		if k in month_into_quarter_sets:
			data['date']= data['date'].apply(lambda x: x+pd.offsets.MonthBegin(-1))
			data = data.groupby('date').mean()
			data.reset_index(inplace=True)
		if k in quarter_sets:
			data['date']= data['date'].apply(lambda x: x+pd.offsets.MonthBegin(2))
			data = data.groupby('date').mean()
			data.reset_index(inplace=True)

		data_df = data_df.merge(data,how='left')

	## Processing ##
