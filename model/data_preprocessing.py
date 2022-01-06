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
# master_df.index = master_df["date"]
# master_df = master_df.drop(labels=["date"], axis=1, inplace=False)
# perform pct change arithmetic
# master_df = master_df.pct_change()
# reset index - again need to better understand
# master_df = master_df.reset_index()
# melt data set to flatten tabular data
master_df = master_df.melt(id_vars= ["date"])
master_df.rename(columns={"date":"year","variable":"MSA"},inplace=True)
#print to csv
df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","all_msa_gdp.csv"), index=False)
df_gdp = master_df

# CPI file prep
for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")):
    if file.endswith("cpi.xlsx"):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data",file)
        full = pd.read_excel(filepath)
        # label columns

        df = pd.DataFrame(full.iloc[2:].values,columns=full.iloc[0].values)
        # define in file where the dataframe is
        # convert date column to date format
        df["date"] = df['date'].apply(lambda x: str(x)[0:4])
        # condense to annual rows and take mean of all monthly values
        df = df.astype(float)
        df = df.groupby("date").mean().reset_index()

        df.index = df["date"]
        df = df.drop(labels=["date"], axis = 1, inplace=False).reset_index()
        
        # df = df.diff().reset_index()
        df = df.melt(id_vars = "date", var_name = "MSA")
        
        df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","all_msa_cpi.csv"), index=False)
        df.rename(columns={"date":"year"},inplace=True)
        df_cpi = df
        
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
        # df = df.diff(axis = 1)
        
        # re-insert index
        df = df.reset_index()
        
        #melt again
        df = df.melt(id_vars = "MSA", var_name = "year")
        df = df[["year", "MSA", "value"]]
        
        # print to csv
        df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","all_msa_cr.csv"), index=False)
        df_cap_rate = df
        
df_cap_rate['metric'] = 'cap_rate'
df_cpi['metric'] = 'cpi'
df_gdp['metric'] = 'gdp'
df = pd.concat([df_cap_rate,df_cpi,df_gdp],axis=0,ignore_index=True)
df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","gdp_cpi_cr_combined.csv"), index=False)
