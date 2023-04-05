import click


@click.group()
def cli_visualize():
    pass


@cli_visualize.command()
@click.option("--ip", type=str, default="0.0.0.0", help="ip address to open tensorboard.")
@click.option("--port", type=int, default=6006, help="port to open tensorboard.")
@click.option("--log_dir", type=str, default=".", help="log dir to open tensorboard.")
def tensorboard(ip, port, log_dir):
    import os
    os.system("tensorboard --logdir " + log_dir + " --host=" + ip + " --port=" + str(port))

if __name__ == '__main__':
    cli_visualize()
