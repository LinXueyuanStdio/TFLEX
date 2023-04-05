import click


@click.group()
def cli_TODO():
    pass


@cli_TODO.command()
@click.option("--output_dir", type=str, default="output", help="Which dir to output.")
def TODO():
    raise NotImplementedError

if __name__ == '__main__':
    cli_TODO()
