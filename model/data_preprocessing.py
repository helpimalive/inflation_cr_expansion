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
# index repair work - need to better understand
master_df.index = master_df["date"]
master_df = master_df.drop(labels=["date"], axis=1, inplace=False)
# perform pct change arithmetic
master_df = master_df.pct_change()
# reset index - again need to better understand
master_df = master_df.reset_index()
# melt data set to flatten tabular data
master_df = master_df.melt(id_vars= "date", var_name="msa")
# get rid of na values
master_df = master_df.dropna()
#print to csv
master_df.to_csv('gdp_output.csv', index=False)

# CPI file prep
for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")):
    if file.endswith("cpi.xlsx"):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data",file)
        df = pd.read_excel(filepath)
        # define in file where the dataframe is
        df = df.iloc[1:,0:20]
        # label columns
        df.columns = ["date","Boston", "Philadelphia", "Chicago", "Dallas", "Houston", "Atlanta", "Miami", 
        "SanFrancisco", "Tampa", "Minneapolis", "StLouis", "Seattle", "Denver", "WashingtonDC", 
        "LosAngeles", "Baltimore", "SanDiego", "Phoenix", "NewYork"]
        # convert date column to date format
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.to_period("Y")
        # condense to annual rows and take mean of all monthly values
        df = df.groupby(["date"]).mean()
        df = df.reset_index()
        df.index = df["date"]
        df = df.drop(labels=["date"], axis = 1, inplace=False)
        
        df = df.pct_change()
        df = df.dropna()
        df = df.reset_index()
        df = df.melt(id_vars = "date", var_name = "msa")
        
df.to_csv('cpi_output.csv', index=False)

        
# Cap Rate file prep
for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")):
    if file.endswith("cr.xlsx"):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data",file)
        df = pd.read_excel(filepath)
        #df = df.transpose()
        # remove duplicative Geography Name row/column
        df = df.drop(labels=["Geography Name"], axis=1, inplace=False)
        # melt to annualize
        df = df.melt(id_vars = "MSA", var_name = "year")
        # correct all years
        df.year = df.year.apply(lambda x: x[1:5])        
        #group by mean to annualize
        df = df.groupby(["MSA","year"]).mean()
        # reset index before pivot
        df = df.reset_index()
        #pivot to set up for difference
        df = df.pivot_table(columns = ["year"],
                            index = ["MSA"])
        
        # diff
        df = df.sort_index(axis=1, level=1).droplevel(0, axis=1)
        df = df.diff(axis = 1)
        
        # re-insert index
        df = df.reset_index()
        
        # drop n/a's
        df = df.dropna(axis = 1)
        
        #melt again
        df = df.melt(id_vars = "MSA", var_name = "year")
        df = df[["year", "MSA", "value"]]
        
        # print to csv
        df.to_csv('cr_output.csv', index=False)
       
        

        
        
     
    
        # reset index to re-introduce MSA column
        #df = df.reset_index()
        # melt
        #df = df.melt(id_vars = "MSA", var_name = "year")
        #df.year = df.year.apply(lambda x: x[1:5])        
        #df = df.groupby(["MSA","year"]).mean()
       #df.index = df["MSA"]
       #df = df.drop("MSA", axis = 1)
       #df = df.diff(axis = 0)


        
print(df.head()) 

#print(list(df.columns.values))
# for file in 

## print to csv
# master_df.to_csv('test_output.csv', index=False)



"""

pandas melt to convert from PIVOT to end goal

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