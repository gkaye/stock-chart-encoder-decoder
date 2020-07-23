import torch
import yfinance as yf
import os
import pandas as pd

# Method to downselect columns on array of pandas dataframes
def downselect_columns(data, columns):
    for i in range(len(data)):
        data[i] = data[i][columns]


# Method to generate a list of tensors a list of pandas dataframes
def generate_tensors(data):
    ret = []
    for d in data:
        ret.append(torch.FloatTensor(d.values))
    return ret


# Method to split a tensor into overlapping chunks of the indicated size or less, torch.split() is not overlapping.
# Returns a list of tensors
def generate_chunks(tensor, chunk_size):
    ret = []
    for lower_bound in range(tensor.size(0) - chunk_size + 1):
        upper_bound_exclusive = lower_bound + chunk_size
        # Note that this slice is a deep copy.  This is critical for
        # future pre-processing of the data
        cloned_slice = tensor[lower_bound:upper_bound_exclusive].clone()
        ret.append(cloned_slice)
    return ret


# generates chunks from list of tensors
def multi_generate_chunks(tensor_list, chunk_size):
    ret = []
    for t in tensor_list:
        ret.extend(generate_chunks(t, chunk_size))
    return ret


# Method to normalize a tensor with shape chunk_size x 5 (ohlcv)
# Normalizes between 0 and 1
def normalize_tensor(tensor):
    ohlc = tensor[:, :-1]
    v = tensor[:, -1]

    # Normalize and set ohlc
    ohlc_range = (ohlc.max() - ohlc.min())
    if ohlc_range != 0:
        tensor[:, :-1] = (ohlc - ohlc.min()) / ohlc_range
    else:
        tensor[:, :-1].fill_(0.5)

    # Normalize and set v
    v_range = (v.max() - v.min())
    if v_range != 0:
        tensor[:, -1] = (v - v.min()) / v_range
    else:
        tensor[:, -1].fill_(0.5)


# Method that normalizes a list of tensors
def multi_normalize_tensors(tensors_list):
    for t in tensors_list:
        normalize_tensor(t)


# Retrieve the indicated stock.  Result is cached, checks are cached ONLY BY ticker name.
def download_stock(ticker, start_date, end_date, interval, dl_dir):
    dl_path = f'{dl_dir}/{ticker}.pkl'
    dl = None

    # Cache
    if os.path.isfile(dl_path):
      print(f'{dl_path} already downloaded, loading from file...')
      dl = pd.read_pickle(dl_path)
    else:
      dl = yf.download(ticker, start_date, end_date, interval = interval)
      os.makedirs(dl_dir, exist_ok=True)
      dl.to_pickle(dl_path)

    return dl


# Downloads all of the indicated stocks.  Results are cached ONLY BY ticker name.
def download_stocks(tickers, start_date, end_date, interval, dl_dir):
    ret = []
    for ticker in tickers:
        ret.append(download_stock(ticker, start_date, end_date, interval, dl_dir))

    return ret
