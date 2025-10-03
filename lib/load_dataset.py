# data_loader.py (updated)
import os
import numpy as np
import pandas as pd

# ----------------- Helpers -----------------
def _ensure_tnc_from_df(df: pd.DataFrame):
    """DataFrame (rows=T, cols=N) -> (T, N, 1) float32"""
    x = df.to_numpy().astype(np.float32)    # (T, N)
    x = np.expand_dims(x, axis=-1)          # (T, N, 1)
    return x

def _read_h5_df(path, key='/df'):
    if not os.path.exists(path):
        raise FileNotFoundError(f'H5 not found: {path}')
    with pd.HDFStore(path, mode='r') as store:
        keys = store.keys()
        if key not in keys:
            if len(keys) == 0:
                raise ValueError(f'No keys in H5: {path}')
            key = keys[0]
        df = store[key]                     # rows=T (datetime?), cols=sensors
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f'Index is not DatetimeIndex and cannot be parsed: {e}')
    return df

def _tod_from_datetime_index(dt_index: pd.DatetimeIndex, num_nodes: int):
    """Time-of-day ∈ [0,1): minute_of_day / 1440 → (T, N, 1)"""
    mins = (dt_index.hour * 60 + dt_index.minute).astype(np.int64)      # (T,)
    tod  = (mins.astype(np.float32) / (24*60))[:, None]                  # (T,1)
    return np.repeat(tod, num_nodes, axis=1)[..., None].astype(np.float32)  # (T,N,1)

def _dow_from_datetime_index(dt_index: pd.DatetimeIndex, num_nodes: int):
    """
    Day-of-Week نرمال‌شده در [0,1] (Mon=0 ... Sun=6 → /6).
    خروجی: (T, N, 1)
    """
    dow = (dt_index.weekday.values.astype(np.float32) / 6.0)[:, None]   # (T,1)
    return np.repeat(dow, num_nodes, axis=1)[..., None].astype(np.float32)  # (T,N,1)

def maybe_add_time_channels(base: np.ndarray,
                            *,
                            add_tod: bool = False,
                            add_dow: bool = False,
                            dt_index: pd.DatetimeIndex | None = None):
    """
    base: (T, N, C>=1)
    اگر add_tod / add_dow True باشد، کانال متناظر اضافه می‌شود.
    اگر dt_index داده شده باشد از آن استفاده می‌کنیم، وگرنه از قدم‌های 5 دقیقه‌ای می‌سازیم.
    (در این نسخه فقط برای METR-LA/PEMS-BAY استفاده می‌کنیم)
    """
    if not (add_tod or add_dow):
        return base

    T, N, _ = base.shape
    extras = []

    if dt_index is not None:
        if add_tod: extras.append(_tod_from_datetime_index(dt_index, N))
        if add_dow: extras.append(_dow_from_datetime_index(dt_index, N))
    else:
        # ساخت synthetic (اگر لازم شد)، ولی این فایل فعلاً فقط H5ها را هدف می‌گیرد.
        idx = np.arange(T, dtype=np.int64)
        if add_tod:
            tod = ((idx % 288) / 288.0).astype(np.float32)[:, None]
            tod = np.repeat(tod, N, axis=1)[..., None]
            extras.append(tod)
        if add_dow:
            dow = (((idx // 288) % 7) / 6.0).astype(np.float32)[:, None]
            dow = np.repeat(dow, N, axis=1)[..., None]
            extras.append(dow)

    return np.concatenate([base] + extras, axis=-1) if extras else base


# ----------------- Main Loader -----------------
def load_st_dataset(dataset, add_tod: bool = False, add_dow: bool = False):
    """
    Returns data with shape (T, N, C).
    - فقط METR-LA و PEMS-BAY را با کانال‌های زمان (براساس فلگ‌ها) آپدیت می‌کنیم.
    - PEMSD3/4/7/8 دست‌نخورده می‌مانند (فقط سیگنال کانال 0).
    - کانال 0 همیشه سیگنال اصلی است.
    """
    if dataset == 'PEMSD4':
        data = np.load('/content/CaST-GCRN/data/PEMS04/pems04.npz')['data'][:, :, 0]
        if data.ndim == 2: data = data[..., None]
        data = data.astype(np.float32)      # (T, N, 1)
        print('Load %s:' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
        return data

    elif dataset == 'PEMSD8':
        data = np.load('/content/CaST-GCRN/data/PEMS08/pems08.npz')['data'][:, :, 0]
        if data.ndim == 2: data = data[..., None]
        data = data.astype(np.float32)
        print('Load %s:' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
        return data

    elif dataset == 'PEMSD3':
        data = np.load('/content/CaST-GCRN/data/PEMS03/PEMS03.npz')['data'][:, :, 0]
        if data.ndim == 2: data = data[..., None]
        data = data.astype(np.float32)
        print('Load %s:' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
        return data

    elif dataset == 'PEMSD7':
        data = np.load('/content/CaST-GCRN/data/PEMS07/PEMS07.npz')['data'][:, :, 0]
        if data.ndim == 2: data = data[..., None]
        data = data.astype(np.float32)
        print('Load %s:' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
        return data

    elif dataset == 'METR-LA':
        h5_path = '/content/CaST-GCRN/data/METR-LA/metr-la.h5'
        df = _read_h5_df(h5_path, key='/df')                    # rows=T, cols=N=207
        speed = _ensure_tnc_from_df(df)                         # (T, N, 1)
        # فقط اینجا به‌صورت کنترل‌شده کانال‌های زمان رو اضافه می‌کنیم
        data  = maybe_add_time_channels(speed, add_tod=add_tod, add_dow=add_dow, dt_index=df.index)
        print('Load %s:' % dataset, data.shape, float(np.max(data)), float(np.min(data)),
              float(np.mean(data)), float(np.median(data)))
        return data

    elif dataset == 'PEMS-BAY':
        h5_path = '/content/CaST-GCRN/data/PEMS-BAY/pems-bay.h5'
        df = _read_h5_df(h5_path, key='/df')                    # rows=T, cols=N=325
        speed = _ensure_tnc_from_df(df)                         # (T, N, 1)
        data  = maybe_add_time_channels(speed, add_tod=add_tod, add_dow=add_dow, dt_index=df.index)
        print('Load %s:' % dataset, data.shape, float(np.max(data)), float(np.min(data)),
              float(np.mean(data)), float(np.median(data)))
        return data

    else:
        raise ValueError(f'Unknown dataset: {dataset}')
