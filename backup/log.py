"""
@date: 2021/10/27
@description: null
"""
import inspect
from collections import defaultdict


def m(a, b, *args, **kwargs):
    print(inspect.getfullargspec(m))
    print(args)
    print(kwargs)


if __name__ == "__main__":
    m(1, 2, 6, 3, 4, 5, sdf=2, sfd=3, dfds=5)
    scope = "Test"
    step_num = 10

    row_results = defaultdict(list)
    header = "{0:8s}".format(scope)
    row_results[header].append("avg")
    average_metrics = {
        "hits@1": 0.001,
        "hits@5": 0.01,
        "hits@10": 0.1,
        "long_long_long": 4,
    }
    for row in average_metrics:
        cell = average_metrics[row]
        row_results[row].append(cell)
    result = {
        "1p_complex": average_metrics,
        "2p_complex": average_metrics,
        "long_complex": average_metrics,
    }
    query_name_dict = {
        "1p_complex": "1p",
        "2p_complex": "2p",
        "long_complex": "2p_long_long_long",
    }
    for col in result:
        row_results[header].append(query_name_dict[col])
        col_data = result[col]
        for row in col_data:
            cell = col_data[row]
            row_results[row].append(cell)
    print("step: %d" % step_num)


    def to_str(data):
        if isinstance(data, float):
            return "{0:>6.2%}  ".format(data)
        elif isinstance(data, int):
            return "{0:^6d}  ".format(data)
        else:
            return "{0:^6s}  ".format(data[:6])


    for i in row_results:
        row = row_results[i]
        print("{0:<8s}".format(i)[:8] + ": " + "".join([to_str(data) for data in row]))
