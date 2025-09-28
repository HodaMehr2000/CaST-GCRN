# import os
# import numpy as np
# import pandas as pd

# def _ensure_tnc_from_df(df: pd.DataFrame):
#     """DataFrame (rows=T, cols=N) -> (T, N, 1) float32"""
#     x = df.to_numpy().astype(np.float32)    # (T, N)
#     x = np.expand_dims(x, axis=-1)          # (T, N, 1)
#     return x

# def _read_h5_df(path, key='/df'):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f'H5 not found: {path}')
#     with pd.HDFStore(path, mode='r') as store:
#         keys = store.keys()
#         if key not in keys:
#             if len(keys) == 0:
#                 raise ValueError(f'No keys in H5: {path}')
#             key = keys[0]
#         df = store[key]                     # rows=T (datetime), cols=sensors
#     return df

# def load_st_dataset(dataset):
#     """
#     Returns data with shape (T, N, C).
#     """
#     if dataset == 'PEMSD4':
#         data = np.load('/content/CaST-GCRN/data/PEMS04/pems04.npz')['data'][:, :, 0]
#         if data.ndim == 2: data = data[..., None]
#         data = data.astype(np.float32)      # (T, N, 1)
#         print('Load %s:' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
#         return data

#     elif dataset == 'PEMSD8':
#         data = np.load('/content/CaST-GCRN/data/PEMS08/pems08.npz')['data'][:, :, 0]
#         if data.ndim == 2: data = data[..., None]
#         data = data.astype(np.float32)
#         print('Load %s:' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
#         return data

#     elif dataset == 'PEMSD3':
#         data = np.load('/content/CaST-GCRN/data/PEMS03/PEMS03.npz')['data'][:, :, 0]
#         if data.ndim == 2: data = data[..., None]
#         data = data.astype(np.float32)
#         print('Load %s:' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
#         return data

#     elif dataset == 'PEMSD7':
#         data = np.load('/content/CaST-GCRN/data/PEMS07/PEMS07.npz')['data'][:, :, 0]
#         if data.ndim == 2: data = data[..., None]
#         data = data.astype(np.float32)
#         print('Load %s:' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
#         return data

#     elif dataset == 'METR-LA':
#         h5_path = '/content/CaST-GCRN/data/METR-LA/metr-la.h5'  # path
#         df = _read_h5_df(h5_path, key='/df')                    # rows=T, cols=N=207
#         data = _ensure_tnc_from_df(df)                          # (T, 207, 1)
#         print('Load %s:' % dataset, data.shape, float(np.max(data)), float(np.min(data)),
#               float(np.mean(data)), float(np.median(data)))
#         return data

#     elif dataset == 'PEMS-BAY':
#         h5_path = '/content/CaST-GCRN/data/PEMS-BAY/pems-bay.h5'
#         df = _read_h5_df(h5_path, key='/df')                    # rows=T, cols=N=325
#         data = _ensure_tnc_from_df(df)                          # (T, 325, 1)
#         print('Load %s:' % dataset, data.shape, float(np.max(data)), float(np.min(data)),
#               float(np.mean(data)), float(np.median(data)))
#         return data

#     else:
#         raise ValueError(f'Unknown dataset: {dataset}')



import os
import numpy as np
import pandas as pd

# --- تنظیم ساده برای روشن/خاموش کردن DOW بدون تغییر امضای تابع ---
ADD_DOW_FOR_H5 = True   # اگر نخواستی DOW اضافه شود، False کن

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
        df = store[key]                     # rows=T (datetime), cols=sensors
    return df

def _dow_from_datetime_index(dt_index: pd.DatetimeIndex, num_nodes: int):
    """
    Day-of-Week عددی نرمال‌شده در بازه [0,1] (Mon=0 ... Sun=6 -> /6).
    خروجی: (T, N, 1)
    """
    # .weekday: Monday=0 ... Sunday=6
    dow = (dt_index.weekday.values.astype(np.float32) / 6.0)[:, None]  # (T,1)
    return np.tile(dow, (1, num_nodes))[:, :, None]                    # (T,N,1)

def load_st_dataset(dataset):
    """
    Returns data with shape (T, N, C).
    برای METR-LA و PEMS-BAY (H5 با index زمانی)، در صورت فعال بودن ADD_DOW_FOR_H5
    کانال DOW به کانال سرعت اضافه می‌شود → C=2. برای PEMSهای NPZ، C=1 می‌ماند.
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
        if ADD_DOW_FOR_H5:
            dow = _dow_from_datetime_index(df.index, speed.shape[1])  # (T, N, 1)
            data = np.concatenate([speed, dow], axis=-1)        # (T, N, 2)
        else:
            data = speed                                        # (T, N, 1)
        print('Load %s:' % dataset, data.shape, float(np.max(data)), float(np.min(data)),
              float(np.mean(data)), float(np.median(data)))
        return data

    elif dataset == 'PEMS-BAY':
        h5_path = '/content/CaST-GCRN/data/PEMS-BAY/pems-bay.h5'
        df = _read_h5_df(h5_path, key='/df')                    # rows=T, cols=N=325
        speed = _ensure_tnc_from_df(df)                         # (T, N, 1)
        if ADD_DOW_FOR_H5:
            dow = _dow_from_datetime_index(df.index, speed.shape[1])  # (T, N, 1)
            data = np.concatenate([speed, dow], axis=-1)        # (T, N, 2)
        else:
            data = speed
        print('Load %s:' % dataset, data.shape, float(np.max(data)), float(np.min(data)),
              float(np.mean(data)), float(np.median(data)))
        return data

    else:
        raise ValueError(f'Unknown dataset: {dataset}')
