import os
import numpy as np
import pandas as pd

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

def load_st_dataset(dataset):
    """
    Returns data with shape (T, N, C).
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
        h5_path = '/content/CaST-GCRN/data/METR-LA/metr-la.h5'  # path
        df = _read_h5_df(h5_path, key='/df')                    # rows=T, cols=N=207
        data = _ensure_tnc_from_df(df)                          # (T, 207, 1)
        print('Load %s:' % dataset, data.shape, float(np.max(data)), float(np.min(data)),
              float(np.mean(data)), float(np.median(data)))
        return data

    elif dataset == 'PEMS-BAY':
        h5_path = '/content/CaST-GCRN/data/PEMS-BAY/pems-bay.h5'
        df = _read_h5_df(h5_path, key='/df')                    # rows=T, cols=N=325
        data = _ensure_tnc_from_df(df)                          # (T, 325, 1)
        print('Load %s:' % dataset, data.shape, float(np.max(data)), float(np.min(data)),
              float(np.mean(data)), float(np.median(data)))
        return data

    else:
        raise ValueError(f'Unknown dataset: {dataset}')



# import os
# import numpy as np

# def load_st_dataset(dataset):
#     #output B, N, D
#     if dataset == 'PEMSD4':
#         #data_path = os.path.join('data/PeMSD4/pems04.npz')
#         data = np.load("/content/CaST-GCRN/data/PEMS04/pems04.npz")['data'][:, :, 0]  #onley the first dimension, traffic flow data
#         #data = np.load("C:/Users/Hoda/A - Uni/thesis/SGCRN_RD/data/PEMS04/pems04.npz")['data'][:, :, 0]  #onley the first dimension, traffic flow data
#     elif dataset == 'PEMSD8':
#         #data_path = os.path.join('data/PeMSD8/pems08.npz')
#         data = np.load('/content/CaST-GCRN/data/PEMS08/pems08.npz')['data'][:, :, 0]  #onley the first dimension, traffic flow data

#     elif dataset == 'PEMSD3':
#         #data_path = os.path.join('data/PeMSD8/pems08.npz')
#         data = np.load('/content/CaST-GCRN/data/PEMS03/PEMS03.npz')['data'][:, :, 0]  #onley the first dimension, traffic flow data    

#     elif dataset == 'PEMSD7':
#         #data_path = os.path.join('data/PeMSD8/pems08.npz')
#         data = np.load('/content/CaST-GCRN/data/PEMS07/PEMS07.npz')['data'][:, :, 0]  #onley the first dimension, traffic flow data 

#     else:
#         raise ValueError
#     if len(data.shape) == 2:
#         data = np.expand_dims(data, axis=-1)
#     print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
#     return data 
