import json
import pandas as pd

from pathlib import Path


def load_annotations(path: str) -> pd.DataFrame:
    """
    Loads a csv-like annotation file with fields ["frame", "ID", "left", "top",
    "width", "height", "confidence", "null1", "null2", "null3"] into a pandas
    dataframe. Check Nvidia AICity challenge readme for further detail.

    Parameters
    ----------
    path: str
        Path string for the input file.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe which contains the loaded csv with only the needed
        columns ["frame", "ID", "left", "top", "width", "height", "confidence"].
    """
    ann = pd.read_csv(
        path,
        sep=",",
        names=["frame", "ID", "left", "top", "width", "height", "confidence", "x", "y", "z"]
    )
    return ann
