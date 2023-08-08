"""
@date: 2022/4/25
@description: null
"""
import os

import click

from toolbox.exp.OutputSchema import OutputSchema, Cleaner


@click.group()
def cli_clean():
    pass

@cli_clean.command()
@click.option("--output_dir", type=str, default="output", help="Which dir to clean")
def clean_output(output_dir):
    dirs = os.listdir(output_dir)
    for d in dirs:
        output = OutputSchema(str(d))
        cleaner = Cleaner(output.pathSchema)
        cleaner.remove_non_best_checkpoint_and_model()


if __name__ == '__main__':
    cli_clean()
