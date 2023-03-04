import json
import pickle
from typing import Callable

import torch

from src import *

DATA_FORMATS = {".json": {"load": json.load, "dump": json.dump, "type": ""},
                ".pkl": {"load": pickle.load, "dump": pickle.dump, "type": "b"},
                ".pth": {"load": torch.load, "dump": torch.save, "type": "b"}}

file_ending: Callable[[str | Path], str] = lambda p: Path(p).suffix
handler_for_file_ending: Callable[[str | Path], Callable] = lambda p: DATA_FORMATS[file_ending(p)]
type_for_file_ending: Callable[[str | Path], Callable] = lambda p: DATA_FORMATS[file_ending(p)]["type"]

TO_CACHE_PATH: Callable[[str], Path] = lambda s: CACHE_DATA / s
TO_MODEL_PATH: Callable[[str], Path] = lambda s: MODELS / s


def load(path: Path | str):
    if isinstance(path, Path):
        path = str(path)
    with open(path, mode="r"+type_for_file_ending(path)) as f:
        return handler_for_file_ending(path)["load"](f)


def save(path: Path | str, obj):
    if isinstance(path, Path):
        path = str(path)
    with open(path, mode="w"+type_for_file_ending(path)) as f:
        return handler_for_file_ending(path)["dump"](obj, f)


def save_cached(savename: str, obj: object, overwrite: bool = True):
    savename = TO_CACHE_PATH(savename)
    if savename.is_file() and not overwrite:
        raise FileExistsError(f"File {savename} already exists!")
    save(savename, obj)


def load_cached(savename: str):
    savename = TO_CACHE_PATH(savename)
    if not savename.is_file():
        raise FileNotFoundError(f"File {savename} doesn't exist!")
    return load(savename)


def save_model(savename: str, obj: object, overwrite: bool = True):
    savename = TO_MODEL_PATH(savename)
    if savename.is_file() and not overwrite:
        raise FileExistsError(f"File {savename} already exists!")
    save(savename, obj)


def load_model(savename: str):
    savename = TO_MODEL_PATH(savename)
    if not savename.is_file():
        raise FileNotFoundError(f"File {savename} doesn't exist!")
    return load(savename)


def cache_result(savename: str):
    def _decorator(f):
        def _f(*args, **kwargs):
            try:
                _obj = load_cached(savename)
            except FileNotFoundError:
                _obj = f(*args, **kwargs)
                save_cached(savename, _obj)
            return _obj

        return _f

    return _decorator
