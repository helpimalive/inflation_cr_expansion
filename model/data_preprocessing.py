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
        df["date"] = df["date"].dt.to_period("Y")+1
        # NOTE
        # Here we going to add 1 because the GDPs come out literally a year after the end of the reference year
        # i.e., 2020 data is published 12/8/2021

        if master_df is not None:
            master_df = pd.concat([master_df,df[msa_label]],axis=1)
        else:
            master_df = df

master_df = master_df.melt(id_vars= ["date"])
master_df.rename(columns={"date":"year","variable":"MSA"},inplace=True)
master_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","all_msa_gdp.csv"), index=False)
df_gdp = master_df

# CPI file prep
for file in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")):
    if file.endswith("cpi.xlsx"):
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data",file)
        full = pd.read_excel(filepath)
        df = pd.DataFrame(full.iloc[2:].values,columns=full.iloc[0].values)
        df = df.melt(id_vars = ['date']).dropna()
        df['value'] = df['value'].astype(float)
        df['year'] = df.date.dt.year
        df = df.sort_values(by=['variable','date'],ascending=False)
        df = df.groupby(['variable','year']).head(1).reset_index(drop=True)
        # df = df.pivot_table(columns=['variable'],index=['year'],values=['value']).pct_change().reset_index()
        # df = df.melt(id_vars=['year'],value_name='value_X')
        # df = df.rename(columns={'value_X':'value'})
        df = df[['year','variable','value']]
        # df = df.drop(labels=["date"], axis = 1, inplace=False).reset_index()
        df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","all_msa_cpi.csv"), index=False)
        df.rename(columns={'variable':'MSA'},inplace=True)
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
        df['yyyy'] = df.year.apply(lambda x: x[1:5])        
        df['qq'] = df.year.apply(lambda x: x[:5:])
        df.rename(columns={'year':'date'},inplace=True)
        df = df.sort_values(by=['MSA','date'],ascending=False)
        df = df.groupby(['MSA','yyyy']).head(1).reset_index(drop=True)
        df = df.pivot_table(columns = ["yyyy"],
                            index = ["MSA"])
        df = df.sort_index(axis=1, level=1).droplevel(0, axis=1)
        df = df.reset_index()
        df = df.melt(id_vars = "MSA", var_name = "year")
        df = df[["year", "MSA", "value"]]
        
        # print to csv
        df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","all_msa_cr.csv"), index=False)
        df_cap_rate = df
        
df_cap_rate['metric'] = 'cap_rate'
df_cpi['metric'] = 'cpi'
df_gdp['metric'] = 'gdp'
df = pd.concat([df_cap_rate,df_cpi,df_gdp],axis=0,ignore_index=True)
keep = ['Atlanta',
'Baltimore',
'Boston',
'Chicago',
'Dallas',
'Denver',
'Detroit',
'Houston',
'LosAngeles',
'Miami',
'Minneapolis',
'NewYork',
'Philadelphia',
'Phoenix',
'SanDiego',
'SanFrancisco',
'Seattle',
'StLouis',
'Tampa',
'WashingtonDC'
]
df = df[df['MSA'].isin(keep)]
df.to_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","gdp_cpi_cr_combined.csv"), index=False)
