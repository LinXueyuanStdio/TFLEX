"""
@date: 2021/12/9
@description: 这里维护各种模型的复现结果。用户进行实验后，可以拉取实验结果，一键生成对应的 latex 表格，就不用手动抄写到论文中了
https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html
"""
from typing import *

import pandas as pd


def QueryEmbeddingLeaderboard():
    header = ["Model", "1p", "2p", "3p", "2i", "3i", "pi", "ip", "2u", "up", "AVG"]
    FB15k = [
        ["GQE",    53.9, 15.5, 11.1, 40.2, 52.4, 27.5, 19.4, 22.3, 11.7, 28.2],
        ["Q2B",    70.5, 23.0, 15.1, 61.2, 71.8, 41.8, 28.7, 37.7, 19.0, 40.1],
        ["BetaE",  65.1, 25.7, 24.7, 55.8, 66.5, 43.9, 28.1, 40.1, 25.2, 41.6],
        ["LogicE", 72.3, 29.8, 26.2, 56.1, 66.3, 42.7, 32.6, 43.4, 27.5, 44.1],
        ["ConE",   73.3, 33.8, 29.2, 64.4, 73.7, 50.9, 35.7, 55.7, 31.4, 49.8],
    ]
    FB237 = [
        ["GQE",    35.2, 7.4,  5.5,  23.6, 35.7, 16.7, 10.9, 8.4,  5.8,  16.6],
        ["Q2B",    41.3, 9.9,  7.2,  31.1, 45.4, 21.9, 13.3, 11.9, 8.1,  21.1],
        ["BetaE",  39.0, 10.9, 10.0, 28.8, 42.5, 22.4, 12.6, 12.4, 9.7,  20.9],
        ["LogicE", 41.3, 11.8, 10.4, 31.4, 43.9, 23.8, 14.0, 13.4, 10.2, 22.3],
        ["ConE",   41.8, 12.8, 11.0, 32.6, 47.3, 25.5, 14.0, 14.5, 10.8, 23.4],
        ["BoolE",  43.3, 13.0, 11.0, 34.5, 48.0, 27.0, 16.7, 15.1, 11.2, 24.4],
    ]
    NELL = [
        ["GQE",    33.1, 12.1, 9.9,  27.3, 35.1, 18.5, 14.5, 8.5,  9.0,  18.7],
        ["Q2B",    42.7, 14.5, 11.7, 34.7, 45.8, 23.2, 17.4, 12.0, 10.7, 23.6],
        ["BetaE",  53.0, 13.0, 11.4, 37.6, 47.5, 24.1, 14.3, 12.2, 8.5,  24.6],
        ["LogicE", 58.3, 17.7, 15.4, 40.5, 50.4, 27.3, 19.2, 15.9, 12.7, 28.6],
        ["ConE",   53.1, 16.1, 13.9, 40.0, 50.8, 26.3, 17.5, 15.3, 11.3, 27.2],
    ]
    return header, FB15k, FB237, NELL


def append_to_QueryEmbeddingLeaderboard(name: str, FB15k_result: List[float], FB237_result: List[float], NELL_result: List[float]):
    header, FB15k, FB237, NELL = QueryEmbeddingLeaderboard()
    FB15k.append([name] + FB15k_result)
    FB237.append([name] + FB237_result)
    NELL.append([name] + NELL_result)
    return header, FB15k, FB237, NELL


def QueryEmbeddingLeaderboard_to_latex_table(name: str,
                                             FB15k_result: List[float],
                                             FB237_result: List[float],
                                             NELL_result: List[float],
                                             output_filename: str = "table.tex"):
    "Dataset"


def dataframe_to_latex_table(df: pd.DataFrame, output_filename: str = "table.tex"):
    columns = list(df.columns)
    df.columns = pd.MultiIndex.from_tuples([
        ("Numeric", "Integers"),
        ("Numeric", "Floats"),
        ("Non-Numeric", "Strings")
    ])
    df.index = pd.MultiIndex.from_tuples([
        ("L0", "ix1"), ("L0", "ix2"), ("L1", "ix3")
    ])
    s = df.style.highlight_max(
        props='cellcolor:[HTML]{FFFF00}; color:{red}; itshape:; bfseries:;'
    )
    s.to_latex(
        column_format="rrrrr", position="h", position_float="centering",
        hrules=True, label="table:5", caption="Styled LaTeX Table",
        multirow_align="t", multicol_align="r"
    )
    pass
