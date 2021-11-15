import pandas as pd


# read in file(s)
df = pd.read_excel(r'path to file')

# pivot and perform operation
df = df.pivot_table(columns='whatever',values='whatever',index='whatever')
df = df.dropna()
df = df.pct_change()
df = df.melt(id_vars= 'whatever')

# merge
master_df = pd.merge(left, right,how='left',left_on='cols',right_on='same_cols')


# to_csv
master_df.to_csv(r'filepath')