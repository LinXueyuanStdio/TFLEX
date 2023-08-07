"""
@date: 2022/3/7
@description: null
"""
import click
from ComplexTemporalQueryData import *


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
@click.option("--dataset", type=str, default="ICEWS14", help="Which dataset to use: ICEWS14, ICEWS05_15, GDELT.")
def main(data_home, dataset):
    if dataset == "ICEWS14":
        dataset = ICEWS14(data_home)
    elif dataset == "ICEWS05_15":
        dataset = ICEWS05_15(data_home)
    elif dataset == "GDELT":
        dataset = GDELT(data_home)
    cache = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
    data = TemporalComplexQueryData(dataset, cache_path=cache)
    data.preprocess_data_if_needed()
    data.patch()


if __name__ == '__main__':
    main()
