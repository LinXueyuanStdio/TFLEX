"""
@date: 2022/3/7
@description: null
"""
import click
from ComplexTemporalQueryData import *


@click.command()
@click.option("--data_home", type=str, default="data", help="The folder path to dataset.")
def main(data_home):
    queries_count = defaultdict(list)
    avg_answers_count = defaultdict(list)
    for dataset in [ICEWS14(data_home), ICEWS05_15(data_home), GDELT(data_home)]:
        cache = ComplexTemporalQueryDatasetCachePath(dataset.cache_path)
        data = TemporalComplexQueryData(dataset, cache_path=cache)
        data.preprocess_data_if_needed()
        data.load_cache(["meta"])
        for k in groups:
            for query_structure in groups[k]:
                v = data.query_meta[query_structure]
                queries_count[query_structure].extend(
                    [str(v["train"]["queries_count"] if v["train"]["queries_count"] > 0 else "-"),
                     str(v["valid"]["queries_count"]),
                     str(v["test"]["queries_count"])])
                avg_answers_count[query_structure].extend(
                    ["{:.2f}".format(v["train"]["avg_answers_count"]),
                     "{:.2f}".format(v["valid"]["avg_answers_count"]),
                     "{:.2f}".format(v["test"]["avg_answers_count"])])
    print("queries_count")
    for k in groups:
        for query_structure in groups[k]:
            print(f"{query_structure} & " + " & ".join(queries_count[query_structure]))

    print("---"*10)
    print("avg_answers_count")
    for k in groups:
        for query_structure in groups[k]:
            print(f"{query_structure} & " + " & ".join(avg_answers_count[query_structure]))


if __name__ == '__main__':
    main()
