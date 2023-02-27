import os
from os.path import join

from tqdm import tqdm

MIN_EPOCH_NUM = 400

if __name__ == '__main__':
    for model_name in tqdm(os.listdir(join('..', 'output', 'keras'))):
        if 'old' in model_name:
            continue
        kept_epochs = []
        snapshots = join('..', 'output', 'keras', model_name, 'snapshots')
        files = os.listdir(snapshots)
        print(snapshots)
        for file in os.listdir(snapshots):
            epoch_num = int(file[file.rfind('_') + 1: file.rfind('.hdf5')])
            if epoch_num < MIN_EPOCH_NUM:
                os.remove(join(snapshots, file))
            elif epoch_num not in kept_epochs:
                kept_epochs.append(epoch_num)
            else:
                os.remove(join(snapshots, file))
                