from full_fred.fred import Fred
import pandas as pd
from datetime import datetime
#from dateutil.relativedata import relativedata

def pull_data():

	API_key = '9e83bfc92bbab5d0086aa294a01fbbae'
	fred 	= Fred(r"C:\Users\amcgrady\Documents\GitHub\inflation_cr_expansion\model\APIkey.txt")
	#go to fred cite for id

	data_dict 													= {}
	data_dict['Gross Domestic Product']							='GDP'
	data_dict['Gross Domestic Product Philadelphia Camden Wilmington PA NJ DE MD']     ='RGMP37980'
	data_dict['Gross Domestic Product Dallas Fort Worth Arlington TX']                 ='RGMP19100'
	data_dict['Gross Domestic Product Houston The Woodlands Sugar Land TX']            ='RGMP26420'
	data_dict['Gross Domestic Product Phoenix Mesa Scottsdale AZ']                     ='RGMP38060'
	data_dict['Gross Domestic Product Denver Aurora Lakewood CO']                      ='RGMP19740'
	data_dict['Gross Domestic Product Washington Arlington Alexandria DC VA MD WV']    ='RGMP47900'
	data_dict['Gross Domestic Product Seattle Tacoma Bellevue WA']                     ='RGMP42660'
	data_dict['Gross Domestic Product New York Newark Jersey City NY NJ PA']           ='RGMP35620'
	data_dict['Gross Domestic Product San Francisco Oakland Hayward CA']               ='RGMP41860'
	data_dict['Gross Domestic Product Los Angeles Long Beach Anaheim CA']              ='RGMP31080'
	data_dict['Gross Domestic Product Chicago Naperville Elgin IL IN WI']              ='RGMP16980'
	data_dict['Gross Domestic Product Tampa St.Petersburg Clearwater FL']              ='RGMP45300'
	data_dict['Gross Domestic Product Atlanta Sandy Spirngs Roswell GA']               ='RGMP12060'
	data_dict['Gross Domestic Product Miami For Lauderdale West Palm Beach FL']        ='RGMP33100'
	data_dict['Gross Domestic Product Boston Cambridge Newton MA NH']                  ='RGMP14460'
	data_dict['Gross Domestic Product San Diego Carlsbad CA']                          ='RGMP41740'
	data_dict['Gross Domestic Product Detroit Warren Dearborn MI']                     ='RGMP19820'
	data_dict['Gross Domestic Product St.Louis Mo IL']                                 ='RGMP41180'
	data_dict['Gross Domestic Product Minneapolis St.Paul Bloomington MN WI']          ='RGMP33460'
	data_dict['Consumer Price Index Philadelphia Camden Wilmington PA NJ DE MD']       ='CUURA102SA0'
	data_dict['Consumer Price Index Dallas Fort Worth Arlington TX']                   ='CUURA316SA0'
	data_dict['Consumer Price Index Houston The Woodlands Sugar Land TX']              ='CUURA318SA0'
	data_dict['Consumer Price Index Phoenix Mesa Scottsdale AZ']                       ='CUUSA429SA0'
	data_dict['Consumer Price Index Denver Aurora Lakewood CO']                        ='CUUSA433SEHA'
	#data_dict['Consumer Price Index Washington Arlington Alexandria DC VA MD WV']      ='CUURS35ASA0'                        
	data_dict['Consumer Price Index Seattle Tacoma Bellevue WA']                       ='CUURA423SEHA'
	data_dict['Consumer Price Index New York Newark Jersey City NY NJ PA']             ='CUURA101SA0'
	data_dict['Consumer Price Index San Francisco Oakland Hayward CA']                 ='CUURA422SA0'
	#data_dict['Consumer Price Index Los Angeles Long Beach Anaheim CA']                ='CUURS49ASA0'
	data_dict['Consumer Price Index Chicago Naperville Elgin IL IN WI']                ='CUURA207SA0'
	data_dict['Consumer Price Index Tampa St.Petersburg Clearwater FL']                ='CUUSA321SA0'
	data_dict['Consumer Price Index Atlanta Sandy Spirngs Roswell GA']                 ='CUURA319SA0'
	data_dict['Consumer Price Index Miami For Lauderdale West Palm Beach FL']          ='CUURA320SA0'
	data_dict['Consumer Price Index Boston Cambridge Newton MA NH']                    ='CUUSA103SA0'
	data_dict['Consumer Price Index San Diego Carlsbad CA']                            ='CUUSA424SA0'
	data_dict['Consumer Price Index Detroit Warren Dearborn MI']                       ='CUURA208SA0'
	data_dict['Consumer Price Index St.Louis Mo IL']                                   ='CUUSA209SA0'
	data_dict['Consumer Price Index Minneapolis St.Paul Bloomington MN WI']            ='CUUSA211SA0'

	

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
	
		
		print(data_df.head(1000))
	
def main():
	pull_data()

if __name__ == '__main__':
	main()

	## Processing ##