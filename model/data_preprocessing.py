""" goal is to clean raw data from RealPage (cap rates), BLS (CPI) and FRED (MSA GDP) and prepare it as y-o-y growth analysis
in CSV format for use in next step in the broader cap rate / inflation study """

import os
import pandas as pd
import re

# read in file(s)
# this loops through all files in data and attempts to read and join them
master_df = None

# GDP files prep
for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")):
    if file.endswith("GDP.xlsx"):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data",file)
        df = pd.read_excel(filepath)
        # define in file where the dataframe is
        df = df.iloc[10:,0:2]
        # label columns
        msa_label= str(re.split('_+',file)[0])
        df.columns = ["date", msa_label]
        #convert date to year
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.to_period("Y")
        # combine all MSA's together, taking only the GDP values for all df's after the first
        if master_df is not None:
            master_df = pd.concat([master_df,df[msa_label]],axis=1)
        else:
            master_df = df    

master_df.to_csv('test_output.csv', index=False)

# CPI file prep
for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")):
    if file.endswith("cpi.xlsx"):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data",file)
        df = pd.read_excel(filepath)
        # define in file where the dataframe is
        df = df.iloc[1:,0:20]
        # label columns
        df.columns = ["Date","Boston", "Philadelphia", "Chicago", "Dallas", "Houston", "Atlanta", "Miami", 
        "SanFrancisco", "Tampa", "Minneapolis", "StLouis", "Seattle", "Denver", "WashingtonDC", 
        "LosAngeles", "Baltimore", "SanDiego", "Phoenix", "NewYork"]
        # convert date column to date format
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] = df["Date"].dt.to_period("Y")
        # condense to annual rows and take mean of all monthly values
        df = df.groupby(["Date"]).mean()
        
# Cap Rate file prep
for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")):
    if file.endswith("cr.xlsx"):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data",file)
        df = pd.read_excel(filepath)
        

# print df to csv        
#df.to_csv('cpi_output.csv', index=True)
        
print(df.head()) 

#print(list(df.columns.values))
# for file in 

## print to csv
# master_df.to_csv('test_output.csv', index=False)



"""
#figure out how to name just MSA df["MSA"] = file
       # if master_df:
           # master_df = master_df.join(df,how="left",left_on=date,right_on=date)
        
# pivot and perform operation of changing on the files
# first we will pivot
# then we will perform changing on the file
df = df.pivot_table(columns='whatever',values='whatever',index='whatever')
df = df.dropna()
df = df.pct_change()
df = df.melt(id_vars= 'whatever')

# merge three files together
master_df = pd.merge(left, right,how='left',left_on='cols',right_on='same_cols')


# write to_csv
master_df.to_csv(r'filepath') """