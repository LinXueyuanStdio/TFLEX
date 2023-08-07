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
    for dataset in [ICEWS14(data_home), ICEWS05_15(data_home), GDELT(data_home)]:
        cache = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
        data = TemporalComplexQueryData(dataset, cache_path=cache)
        data.preprocess_data_if_needed()
        data.load_cache(["meta"])
        for i in data.dump():
            print(i)
        # data.patch2()
        data.patch3()
        del data
        del cache


if __name__ == '__main__':
    main()
