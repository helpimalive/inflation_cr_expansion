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


def _get_fred_data(namespace):
    """Retrieve CPI and GDP data from FRED using API"""
    pass


def _save_fred_data(data, namespace):
    """Save the FRED data as pickle file in the location named by namespace"""
    pass


def _load_fred_data(namespace):
    """Retrieving FRED data from pickle file in namespace location"""
    with _exit_on_error("invalid_path", Exception):
        with open(namespace.input_path, "rb") as f:
            df = pickle.load(f)


def _load_cr_data(namespace):
    """Load Excel file containing Office and MF Cap Rates"""
    with _exit_on_error("file_path", Exception):
        cr_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","green_street_cr_data.xlsx")

    with _exit_on_error("read_in", Exception):
        cr_data = pd.read_excel(cr_path)


    """
    for andrew:
        produce a clean dataframe where the columns of the dataframe match the columns of the excel [Apartment...Office....]
        contain only as many rows as there are dates
            
    v_print(
        name="cr_data",
        value=cr_data,
        namespace=namespace,
    )



def _produce_trailing_avgs(data, namespace):
    """Produce trailing averages n periods for each FRED series"""
    pass


def _generate_signal(data, namespace):
    """Create a signal if conditions are fulfilled in the GDP and CPI series"""
    pass


def _analyze_accuracy(data, namespace):
    """Analyze variance bias tradeoff in estimations of cap rate expansions"""
    pass


def _ingest(namespace):
    """Run import functions"""
    _load_cr_data(namespace)


def _analyze(parser, args):
    """Run analysis functions"""
    namespace = parser.parse_args(args)


########################
#   Argument Adders    #
########################


def _add_look_back_flag(parser):
    """Add the look-back periods to the parser."""
    parser.add_argument(
        "-b",
        "--look-back",
        type=int,
        default=100,
        help="the periods to use to create the groupings",
    )


def _add_hold_pers_flag(parser):
    """Add the hold-periods to the parser."""
    parser.add_argument(
        "-p",
        "--hold-pers",
        type=int,
        default=100,
        help="the number of periods to hold the recommendations",
    )


def _add_start_flag(parser):
    """Add the date to start the analysis to the parser"""
    parser.add_argument(
        "-s",
        "--start-date",
        type=datetime.date.fromisoformat,
        default=None,
        help="the date to start the analysis",
    )

def _add_input_path_ingest(parser):
    """Add the input path to where the CR data resides"""
    parser.add_argument(
        "-cr",
        "--cap-rate",
        type = pathlib.Path,
        default = os.getcwd(),
        help="the input path to the CR data")


def _add_output_path_ingest(parser):
    """Add the output path to where the fred and cr data resides"""
    parser.add_argument(
        "-o",
        "--output",
        type = pathlib.Path,
        default = os.getcwd(),
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


def _parse_ingest(parser, args):
    with _exit_on_error("unknown", Exception):
        _add_input_path_ingest(parser)
        _add_output_path_ingest(parser)
        _add_verbose_flag(parser)
        namespace = parser.parse_args(args)
        _ingest(namespace)


def _parse_analyze(parser, args):
    namespace = parser.parse_args(args)
    _analyze(namespace)


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
                os.path.join(kwargs["namespace"].output_path, name + ".csv"),
            )
    else:
        kwargs["value"].to_csv(
            os.path.join(kwargs["namespace"].output_path, kwargs["name"] + ".csv"),
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

    ingest = _CommandPair(_parse_ingest, "ingest, clean, and store FRED data")
    analyze = _CommandPair(_parse_analyze, "analyze logic for cap rate expansion")


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
