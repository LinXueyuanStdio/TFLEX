#!/usr/bin/env python3
"""
Main Freqtrade bot script.
Read the documentation to know what cli arguments you need.
"""
import sys

# check min. python version
if sys.version_info < (3, 8):  # pragma: no cover
    sys.exit("Freqtrade requires Python version >= 3.8")

from toolbox import __version__
import click
from toolbox.commands.cli_clean import cli_clean
from toolbox.commands.cli_visualize import cli_visualize

main = click.CommandCollection(sources=[
    cli_clean,
    cli_visualize,
])

if __name__ == '__main__':
    main()
