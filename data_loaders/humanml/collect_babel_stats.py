from data_loaders.get_data import get_dataset_loader
import numpy as np
from tqdm import tqdm
import os

def run():
    # instance dataset
    batch_size = 128
    num_frames = 200
    num_features = 135
    loader = get_dataset_loader('babel', batch_size, num_frames, split='train')

    # all_frames = np.zeros((0, 135), dtype=np.float32)
    all_frames = []

    print('Collecting frames...')
    for motion, args in tqdm(loader):
        assert motion.shape[1] == num_features
        for seq_i, l in enumerate(args['y']['lengths']):
            seq = motion[[seq_i], :, :, :l]
            frames = seq.permute(0, 2, 3, 1).reshape((-1, num_features)).cpu().numpy()
            # all_frames = np.concatenate((all_frames, frames), axis=0)
            all_frames.append(frames)

    all_frames = np.concatenate(all_frames, axis=0)
    _mean = np.mean(all_frames, axis=0)
    _std = np.std(all_frames, axis=0)

    os.makedirs(os.path.join(os.getcwd(), 'babel', 'motion1', 'meta'))
    np.save(os.path.join(os.getcwd(), 'babel', 'motion1', 'meta', 'mean.npy'), _mean)
    np.save(os.path.join(os.getcwd(), 'babel', 'motion1', 'meta', 'std.npy'), _std)

if __name__ == '__main__':
    run()