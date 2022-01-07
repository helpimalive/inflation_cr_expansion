import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

df = pd.read_csv(
    "C:\\Users\\matth\\Documents\\GitHub"
    "\\inflation_cr_expansion\\data\\gdp_cpi_cr_combined.csv"
)

results = pd.DataFrame(
    columns=[
        "mean_pers_one",
        "mean_pers_two",
        "consec_pers",
        "pct_delta_pers_one",
        "pct_delta_pers_two",
        "true_pos",
        "false_negatives",
        "capture",
        "pval",
    ]
)
for mean_pers_one in np.arange(1, 2):
    for mean_pers_two in np.arange(4, 5):
        for consec_pers in np.arange(1, 2):
            for pct_delta_pers_one in np.arange(4, 5):
                for pct_delta_pers_two in np.arange(0, 1):

                    df_cpi = df[df["metric"] == "cpi"]
                    df_cpi = df_cpi.pivot(index="year", columns="MSA", values="value")
                    df_cpi = df_cpi.pct_change()
                    df_mean = df_cpi.rolling(mean_pers_one).mean()
                    df_comp = df_cpi > df_mean.shift(pct_delta_pers_one)
                    df_consec = df_comp.rolling(consec_pers).sum() == (consec_pers)
                    df_cpi_flag = df_consec[~df_mean.isna()]

                    df_gdp = df[df["metric"] == "gdp"]
                    df_gdp = df_gdp.pivot(index="year", columns="MSA", values="value")
                    df_gdp = df_gdp.pct_change()
                    df_mean = df_gdp.rolling(mean_pers_two).mean()

                    df_comp = df_gdp < df_mean.shift(pct_delta_pers_two)
                    df_consec = df_comp.rolling(consec_pers).sum() == (consec_pers)
                    df_gdp_flag = df_consec[~df_mean.isna()]
                    df_flag = (df_gdp_flag == 1) & (df_cpi_flag == 1)

                    df_cr = df[df["metric"] == "cap_rate"]
                    df_cr = df_cr.pivot(index="year", columns="MSA", values="value")
                    df_cr = df_cr.diff()
                    df_cr = df_cr > 0

                    # Uncomment to consider a cap rate expansion one that happens in either
                    # of the next two years
                    # df_cr = df_cr.rolling(2).sum()
                    df_cr = df_cr.shift(-1)
                    df_cr = df_cr > 0

                    mutual_dates = set(df_flag.dropna().index).intersection(
                        df_cr.dropna().index
                    )
                    mutual_dates = set(mutual_dates).intersection(
                        df_gdp_flag.dropna().index
                    )
                    df_flag = df_flag[df_flag.index.isin(mutual_dates)]
                    df_cr = df_cr[df_cr.index.isin(mutual_dates)]
                    positive_accuracy = df_flag == df_cr
                    positive_accuracy = positive_accuracy[df_flag == True]
                    true_positives = positive_accuracy.sum().sum()
                    total_potential_positives = df_cr == True
                    total_potential_positives = total_potential_positives.sum().sum()
                    total_positive_flags = df_flag.sum().sum()
                    if positive_accuracy.count().sum() != 0:
                        true_positive_rate = true_positives / total_positive_flags
                    else:
                        true_positive_rate = np.nan
                    false_positives = total_positive_flags - true_positives
                    capture = true_positives / total_potential_positives

                    negative_accuracy = df_flag == df_cr
                    negative_accuracy = negative_accuracy[df_flag == False]
                    negative_accuracy = negative_accuracy.astype(bool)
                    false_negatives = negative_accuracy == 0
                    false_negatives = false_negatives.sum().sum()
                    total_potential_negatives = df_cr == False
                    total_potential_negatives = total_potential_negatives.sum().sum()
                    total_negative_flags = df_flag == 0
                    total_negative_flags = total_negative_flags.sum().sum()

                    if negative_accuracy.sum().sum() != 0:
                        false_negative_rate = false_negatives / total_negative_flags
                    else:
                        false_negative_rate = np.nan
                    true_negative = total_negative_flags - false_negatives

                    # A contingency table takes groups and categories and compares
                    # observed values to the values expected if the event were random
                    # in our case:
                    #                     CapRateIncrease[Yes] | CapRateIncrease[No]
                    # CapRateFlag[Yes] |    TruePositive       |     FalsePositive
                    # CapRateFlag[No]  |    FalseNegative      |     TrueNegative
                    obs = np.array(
                        [
                            [true_positives, false_positives],
                            [false_negatives, true_negative],
                        ]
                    )

                    if 0 not in obs:
                        chi2, p, dof, ex = chi2_contingency(obs, correction=False)
                    else:
                        p = np.nan

                    trial = pd.DataFrame(
                        [
                            [
                                mean_pers_one,
                                mean_pers_two,
                                consec_pers,
                                pct_delta_pers_one,
                                pct_delta_pers_two,
                                true_positive_rate,
                                false_negative_rate,
                                capture,
                                p,
                            ]
                        ],
                        columns=results.columns,
                    )
                    results = pd.concat([results, trial])
print(df_flag)
df_cr = df[df["metric"] == "cap_rate"]
df_cr = df_cr.pivot(index="year", columns="MSA", values="value")
print(df_cr)
df_flag.to_csv(r"C:\users\matth\desktop\df_msa_flag.csv")
df_cr.to_csv(r"C:\users\matth\desktop\df_cr.csv")
