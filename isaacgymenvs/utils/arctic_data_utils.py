import numpy as np


def inspect_arctic_data(arctic_data_fn):
    data_dict = np.load(arctic_data_fn , allow_pickle=True).item()
    print(data_dict.keys())
    



# python utils/arctic_data_utils.py
if __name__=='__main__':
    arctic_data_fn = '/cephfs/xueyi/data/arctic_processed_data/processed_seqs/s01/microwave_grab_01.npy'
    inspect_arctic_data(arctic_data_fn )

