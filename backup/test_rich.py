"""
@date: 2022/3/14
@description: null
"""
import random
import time

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table


# my_code = '''
# def iter_first_last(values: Iterable[T]) -> Iterable[Tuple[bool, bool, T]]:
#     """Iterate and generate a tuple with a flag for first and last value."""
#     iter_values = iter(values)
#     try:
#         previous_value = next(iter_values)
#     except StopIteration:
#         return
#     first = True
#     for value in iter_values:
#         yield first, False, previous_value
#         first = False
#         previous_value = value
#     yield first, True, previous_value
# '''
# syntax = Syntax(my_code, "python", theme="monokai", line_numbers=True)
# console = Console()
# console.print(syntax)
# code = "\n".join(list(query_structures.values()))
# console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

def generate_table(logs, name="best", last_best_log=None) -> Table:
    """Make a new table."""
    table = Table()
    table.add_column(name)
    for query_name in logs:
        table.add_column(query_name)

    def to_str(data):
        if isinstance(data, float):
            return "{0:>6.2%}  ".format(data)
        elif isinstance(data, int):
            return "{0:^6d}  ".format(data)
        else:
            return "{0:^6s}  ".format(data)

    first_data = list(logs.values())[0]
    for key in first_data:
        row = []
        for query_name in logs:
            current_data = logs[query_name][key]
            if last_best_log is None:
                row.append(to_str(current_data))
            else:
                last_best_data = last_best_log[query_name][key]
                prefix = "[red]" if last_best_data > current_data else "[green]"
                row.append(prefix + to_str(current_data))
        table.add_row(*row)
    return table


all_tasks = ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2in', '3in', 'inp', 'pin', 'pni', '2u-DNF', '2u-DM', 'up-DNF', 'up-DM']

train_steps = 1000
valid_steps = 100
test_steps = 100

test_steps_progress = Progress(
    TextColumn("  "),
    TimeElapsedColumn(),
    TextColumn("({task.completed}/{task.total})"),
    TextColumn("[bold blue]{task.fields[name]}: {task.percentage:.0f}%"),
    BarColumn(),
    "•", TimeRemainingColumn(),
)
overall_progress = Progress(
    TimeElapsedColumn(),
    TextColumn("({task.completed}/{task.total})"),
    TextColumn("[bold blue]{task.fields[name]}: {task.percentage:.0f}%"),
    BarColumn(),
    "•", TimeRemainingColumn(),
    "•", TextColumn("{task.description}"),
)
progress_group = Group(
    overall_progress,
    test_steps_progress,
)
overall_task_id = overall_progress.add_task("[cyan]Train", total=train_steps, name="Train")

with Live(progress_group):
    for step in range(train_steps):
        # update message on overall progress bar
        top_descr = "[bold #AAAAAA]" + " • ".join([f"{k}: {v}" for k, v in {
            "loss": random.randint(100, 200) / 100,
            "pos_loss": random.randint(100, 200) / 100,
            "neg_loss": random.randint(100, 200) / 100,
        }.items()])
        time.sleep(0.01)
        overall_progress.update(overall_task_id, advance=1, description=top_descr)

        if step % 100 == 99:
            test_steps_task_id = test_steps_progress.add_task("", total=test_steps, name="Valid")
            for step in range(test_steps):
                time.sleep(0.01)
                test_steps_progress.update(test_steps_task_id, advance=1)
            test_steps_progress.update(test_steps_task_id, visible=False)
    overall_progress.update(overall_task_id, description="[bold green]done!")
