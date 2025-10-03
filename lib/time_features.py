# lib/time_features.py
import numpy as np
import pandas as pd

def tod_dow_from_datetime_index(dt_index: pd.DatetimeIndex, N: int):
    mins = (dt_index.view('int64') // 60_000_000_000) % (24*60)
    tod = (mins / (24*60)).astype(np.float32)[:, None]
    tod = np.repeat(tod, N, axis=1)[..., None]
    dow = dt_index.weekday.values.astype(np.int64)[:, None]
    dow = np.repeat(dow, N, axis=1)[..., None].astype(np.float32)
    return tod, dow

def tod_dow_from_steps(T: int, N: int):
    idx = np.arange(T, dtype=np.int64)
    tod = ((idx % 288) / 288.0).astype(np.float32)[:, None]
    tod = np.repeat(tod, N, axis=1)[..., None]
    dow = ((idx // 288) % 7).astype(np.float32)[:, None]
    dow = np.repeat(dow, N, axis=1)[..., None]
    return tod, dow
