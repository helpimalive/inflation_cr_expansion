# create a repo
# create pseudo code
# write code
# run pylint
# run black

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
from scipy.stats import chi2_contingency

__version__ = "0.0.0"


########################
# Abnormal Termination #
########################


_ExitStatusPair = collections.namedtuple("_ExitStatusPair", ("status", "message"))


class _ExitStatus(enum.Enum):

    """The exit statuses."""

    success = _ExitStatusPair(0, "success")
    invocation = _ExitStatusPair(1, "invocation")
    unknown = _ExitStatusPair(2, "unknown error")
    keyboard_interrupt = _ExitStatusPair(3, "keyboard interrupt")
    file_path = _ExitStatusPair(4, "file path invalid")
    read_in = _ExitStatusPair(5, "read in data error")

def _main_parser_usage(parser):
    """Return the usage string for the main parser."""

    return re.sub(
        r"\[command\]",
        "command [argument [argument ...]]",
        parser.format_usage().rstrip(),
    )


def _is_help(args):
    """Return True if help is requested, False otherwise."""
    return "-h" in args or "--help" in args


def _exit_abnormally(name, message):
    """Exit the program with the provided status and message."""
    try:
        print(name, message, file=sys.stderr)
    finally:
        # sys.exit(getattr(_ExitStatus, name).value.status)
        sys.exit(getattr(_ExitStatus, name).value.status)


@contextlib.contextmanager
def _exit_on_error(name, *error_types):
    """Catch a matching error and exit with the provided status."""
    try:
        yield
    except error_types as exception:
        _exit_abnormally(name, exception)


######################
# Core functionality #
######################

def _preprocess_raw(namespace):
    '''To preprocess raw dataframes into a single input'''
    pass

def _load_data(namespace):
    """Load Excel file containing cap rates, CPI, and GDP"""
    with _exit_on_error("file_path", Exception):
        if namespace.msa_or_nation == "msa":
            filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","gdp_cpi_cr_combined.csv")
        elif namespace.msa_or_nation == "national":
            filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","National_CR_GDP_CPI.csv")            
        elif namespace.msa_or_nation == "msa_individual":
            filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","gdp_cpi_cr_combined.csv")
        else:
            raise(Exception,"namespace has a bad flag")

    with _exit_on_error("read_in", Exception):
        preprocessed_dataframe = pd.read_csv(filepath)

    v_print(
        name="raw_data_frame",
        value=preprocessed_dataframe,
        namespace=namespace,
    )
    
    return preprocessed_dataframe

def _execute_msa_cr_flag_algo(df, namespace):
    """Runs the algorithm that decides whether a cap rate expansion will occur in the next year or not"""
    # Takes the output from _load_data, a flat dataframe with CR, CPI, and GDP by MSA
    # Returns cap rates analzed; flags generated; statistical metrics of accuracy
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
    mean_pers_one = 1 
    mean_pers_two = 4
    consec_pers = 1
    pct_delta_pers_one = 4
    pct_delta_pers_two = 0

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

    
    v_print(
        name=["results","df_flag","df_cr","df_cpi_flag","df_gdp_flag"],
        value=[results,df_flag,df_cr,df_cpi_flag,df_gdp_flag],
        namespace=namespace,
    )


    return(df_cr,df_flag,results)

def _execute_msa_individual_cr_flag_algo(df, namespace):
    """Runs the algorithm that decides whether a cap rate expansion will occur in the next year or not"""
    # Takes the output from _load_data, a flat dataframe with CR, CPI, and GDP by MSA
    # Also takes the optimal parameters for mean_pers, consec_pers, etc for each MSA
    # Returns cap rates analzed; flags generated; statistical metrics of accuracy
    results = pd.DataFrame(
    columns=[
        "msa",
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

    params =os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","msa_individual_tuning_params.csv")
    params = pd.read_csv(params)
    master = df.copy()
    df_cr_all = pd.DataFrame()
    df_flag_all = pd.DataFrame()
    for msa in master.MSA.unique():
        mean_pers_one = params[params['MSA']==msa]['mean_pers_one'].values[0]
        mean_pers_two = params[params['MSA']==msa]['mean_pers_two'].values[0]
        consec_pers = params[params['MSA']==msa]['consec_pers'].values[0]
        pct_delta_pers_one = params[params['MSA']==msa]['pct_delta_pers_one'].values[0]
        pct_delta_pers_two = params[params['MSA']==msa]['pct_delta_pers_two'].values[0]

        df = master[master['MSA']==msa]        

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
                [   msa,
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
        df_flag.name=[msa]
        df_flag_all = pd.concat([df_flag_all,df_flag],axis=1).dropna()
        df_cr.name=[msa]
        df_cr_all = pd.concat([df_cr_all,df_cr],axis=1).dropna()

    
    v_print(
        name=["results","df_flag","df_cr"],
        value=[results,df_flag_all,df_cr_all],
        namespace=namespace,
    )


    return(df_cr,df_flag,results)

def _execute_national_cr_flag_algo(df, namespace):
    """Runs the algorithm that decides whether a cap rate expansion will occur in the next year or not"""
    # Takes the output from _load_data, a flat dataframe with CR, CPI, and GDP by MSA
    # Returns a flat dataframe of [msa,date,cap rate,cap_rate_expansion_flag,cap_rate_expansion]
        
    results = pd.DataFrame(
        columns=[
            "mean_pers_one",
            "mean_pers_two",
            "forward_pred",
            "pct_delta_pers_one",
            "pct_delta_pers_two",
            "true_pos",
            "false_negatives",
            "capture",
            "pval",
        ]
    )

    df.index = pd.to_datetime(df['date'])
    df.drop(['year','date'],axis=1,inplace=True)
    mean_pers_one = 2
    mean_pers_two = 5
    forward_pred = 8
    pct_delta_pers_one = 5
    pct_delta_pers_two = 3
    consec_pers = 1

    df_cpi = df['CPI'].pct_change()
    df_mean = df_cpi.rolling(mean_pers_one).mean()
    df_comp = df_cpi > df_mean.shift(pct_delta_pers_one)
    df_consec = df_comp.rolling(consec_pers).sum() == (consec_pers)
    df_cpi_flag = df_consec[~df_mean.isna()]
    df_cpi_flag = df_cpi_flag.iloc[pct_delta_pers_one:]

    df_gdp = df['GDP'].pct_change()
    df_mean = df_gdp.rolling(mean_pers_two).mean()
    df_comp = df_gdp < df_mean.shift(pct_delta_pers_two)
    df_consec = df_comp.rolling(consec_pers).sum() == (consec_pers)
    df_gdp_flag = df_consec[~df_mean.isna()]
    df_gdp_flag = df_gdp_flag.iloc[pct_delta_pers_two:]

    df_flag = (df_gdp_flag == 1) & (df_cpi_flag == 1)

    cr_shift = forward_pred
    df_cr = df['CR'].diff(cr_shift)
    # df_cr = df_cr > 0
    df_cr = df_cr.shift(-cr_shift).dropna()
    df_cr = df_cr > 0

    mutual_dates = set(df_flag.dropna().index).intersection(
        df_cr.dropna().index
    )
    mutual_dates = mutual_dates.intersection(
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
                forward_pred,
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
    
    v_print(
        name=["results","df_flag","df_cr"],
        value=[results,df_flag,df_cr],
        namespace=namespace,
    )
    
    return (results,df_flag,df_cr)


def _analyze_accuracy(data, namespace):
    """Analyze variance bias tradeoff in estimations of cap rate expansions"""

    pass

def _graph_msa_results(namespace):
    """Make graphs of the results of the MSA-level forecasts of cap rate expansions"""
    pass


############################
#   The Command Functions  #
############################

def _analyze(parser, args):
    """Run analysis functions"""
    namespace = parser.parse_args(args)
    preprocessed_dataframe = _load_data(namespace)
    if namespace.msa_or_nation == "msa":
        post_processed_dataframe = _execute_msa_cr_flag_algo(preprocessed_dataframe,namespace)
    if namespace.msa_or_nation == "national":
        post_processed_dataframe = _execute_national_cr_flag_algo(preprocessed_dataframe, namespace)
    if namespace.msa_or_nation == "msa_individual":
        post_processed_dataframe = _execute_msa_individual_cr_flag_algo(preprocessed_dataframe, namespace)

def _graph(parser, args):
    """Run the graph functions"""
    namespace = parser.parse_args(args)


########################
#   Argument Adders    #
########################

def _add_msa_national_flag(parser):
    """Add the flag dictating MSA or National Analysis"""
    parser.add_argument(
        "-mn",
        "--msa-or-nation",
        type = str,
        choices=["msa","national","msa_individual"],
        help = "analyze the MSA or National Cap Rates")

def _add_output_path_ingest(parser):
    """Add the output path to where the fred and cr data resides"""
    parser.add_argument(
        "-o",
        "--output",
        type = pathlib.Path,
        default = os.path.join(os.path.dirname(os.getcwd()),"output"),
        help="the output path to the pickled data")

def _add_verbose_flag(parser):
    """Add the verbose flag argument to the parser."""
    parser.add_argument(
        "-v",
        "--verbose-flag",
        type=str,
        default=None,
        choices=[None, "csv", "terminal", "both"],
        help="a flag for the verbose printer",
    )

########################
# Parser Functionality #
########################

def _parse_analyze(parser, args):
    _add_verbose_flag(parser)
    _add_msa_national_flag(parser)
    _add_output_path_ingest(parser)
    _analyze(parser,args)

def _parse_graph(parser, args):
    pass

def v_print(**kwargs):
    namespace = kwargs["namespace"]
    if namespace.verbose_flag == "csv":
        csv_print(**kwargs)
    elif namespace.verbose_flag == "terminal":
        terminal_print(**kwargs)
    elif namespace.verbose_flag == "both":
        csv_print(**kwargs)
        terminal_print(**kwargs)
    else:
        lambda **kwargs: None


def terminal_print(**kwargs):
    if type(kwargs["name"]) == list:
        print(kwargs)
        for i, name in enumerate(kwargs["name"]):
            print(name, kwargs["value"][i])
    else:
        print(kwargs["name"], kwargs["value"])


def csv_print(**kwargs):
    if type(kwargs["name"]) == list:
        for i, name in enumerate(kwargs["name"]):
            kwargs["value"][i].to_csv(
                os.path.join(kwargs["namespace"].output,kwargs["namespace"].msa_or_nation+"_"+ name + ".csv"),
            )
    else:
        kwargs["value"].to_csv(
            os.path.join(kwargs["namespace"].output,kwargs["namespace"].msa_or_nation+"_"+kwargs["name"] + ".csv"),
        )


####################
# Input Validators #
####################


def _validate_dir_path(dir_path):
    """Validate an ingestion or export path"""
    if not os.path.isdir(dir_path):
        _exit_on_error("invalid_path", KeyError)


class _ExitStatusArgumentParser(argparse.ArgumentParser):
    """An argument parser that adds exit statuses to the epilog."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__exit_statuses = set()
        self.add_exit_status(_ExitStatus.success)
        self.add_exit_status(_ExitStatus.unknown)
        self.add_exit_status(_ExitStatus.invocation)
        self.add_exit_status(_ExitStatus.keyboard_interrupt)

    def add_exit_status(self, exit_status):
        """Add the exit status to a private key on the parser."""
        self.__exit_statuses.add(exit_status)

    def parse_args(self, args, *remaining_args, **kwargs):
        """Parse the args. Must pass in args."""
        if _is_help(args):
            self.epilog = os.linesep.join(
                (
                    "exit statuses:",
                    *(
                        "  {} - {}".format(
                            exit_status.value.status, exit_status.value.message
                        )
                        for exit_status in sorted(
                            self.__exit_statuses, key=operator.attrgetter("value")
                        )
                    ),
                )
            )

        return super().parse_args(args, *remaining_args, **kwargs)


_CommandPair = collections.namedtuple("_CommandPair", ("command", "description"))

class _Command(enum.Enum):
    """The commands"""
    analyze = _CommandPair(_parse_analyze, "analyze logic for cap rate expansion")
    graph = _CommandPair(_parse_graph, "graph the result of the analysis")

def main(args=None):
    """Main function: organize command line data"""
    # with _exit_on_error("unknown", Exception):
    with _exit_on_error("keyboard_interrupt", KeyboardInterrupt):
        parser = argparse.ArgumentParser(
            description=main.__doc__, allow_abbrev=False, add_help=False
        )
        parser.add_argument("-h", "--help", action="store_true")
        parser.add_argument(
            "command",
            choices=tuple(command.name for command in _Command),
            nargs="?",
            metavar="command",
        )

        namespace, remaining_args = parser.parse_known_args(args)
        if namespace.command:
            if namespace.help:
                remaining_args.append("-h")
        else:
            if namespace.help:
                print(
                    os.linesep.join(
                        (
                            _main_parser_usage(parser),
                            "",
                            parser.description,
                            "",
                            "optional_arguments:",
                            "  -h, --help            print this help message",
                            "",
                            "command:",
                            *(
                                "    {:20}{}".format(
                                    command.name, command.value.description
                                )
                                for command in _Command
                            ),
                        )
                    )
                )
                sys.exit()
            else:
                _exit_abnormally("invocation", _main_parser_usage(parser))

        for task in [namespace.command]:
            command = getattr(_Command, task)
            command.value.command(
                _ExitStatusArgumentParser(
                    prog="{} {}".format(parser.prog, command.name),
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    description="{}.".format(
                        command.value.description
                    ),
                    allow_abbrev=False,
                ),
                remaining_args,
            )


if __name__ == "__main__":
    main()
