import os
from copy import deepcopy

from os.path import join as pjoin

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.amass.sampling import FrameSampler
from data_loaders.amass.transforms import SMPLTransform
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml import collect_babel_stats
from data_loaders.humanml.options.train_options import TrainDecompOptions
from data_loaders.humanml.utils.plot_script import *

from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import DecompTrainerV3
from data_loaders.humanml.data.dataset import MotionDatasetV2, BABEL_MotionDatasetV2
from data_loaders.humanml.scripts.motion_process import *
from torch.utils.data import DataLoader
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer, POS_enumerator


def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == '__main__':
    parser = TrainDecompOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.foot_contact_entries = 4
    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        opt.dim_pose = 263
        radius = 4
        fps = 20
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        opt.dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    elif opt.dataset_name == 'babel':
        opt.max_epoch *= 20
        opt.foot_contact_entries = 0
        opt.dim_pose = 135
        opt.min_motion_length = 24 # must be at least window size
        opt.max_motion_length = 480
        # opt.data_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'motion1', 'meta')
        # if not os.path.exists(opt.data_root):
        #     collect_babel_stats.run()
    else:
        raise KeyError('Dataset Does Not Exist')


    movement_enc = MovementConvEncoder(opt.dim_pose - opt.foot_contact_entries, opt.dim_movement_enc_hidden,
                                       opt.dim_movement_latent)
    movement_dec = MovementConvDecoder(opt.dim_movement_latent, opt.dim_movement_dec_hidden, opt.dim_pose)

    all_params = 0
    pc_mov_enc = sum(param.numel() for param in movement_enc.parameters())
    print(movement_enc)
    print("Total parameters of prior net: {}M".format(pc_mov_enc *1. // 1e6))
    all_params += pc_mov_enc

    pc_mov_dec = sum(param.numel() for param in movement_dec.parameters())
    print(movement_dec)
    print("Total parameters of posterior net: {}M".format(pc_mov_dec *1. // 1e6))
    all_params += pc_mov_dec

    trainer = DecompTrainerV3(opt, movement_enc, movement_dec)

    if opt.dataset_name == 'babel':
        train_loader = get_dataset_loader('babel',
                                          batch_size=opt.batch_size, num_frames=480,  # not in use
                                          split='train', load_mode='movement_train', opt=opt)
        val_loader = get_dataset_loader('babel',
                                        batch_size=opt.batch_size, num_frames=480,  # not in use
                                        split='val', load_mode='movement_train', opt=opt)

    else:
        mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'Std.npy'))
        dataset_args = {
            'opt': opt, 'mean': mean, 'std': std,
        }
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
        val_args, train_args = deepcopy(dataset_args), deepcopy(dataset_args)
        train_args.update({'split_file': train_split_file})
        val_args.update({'split_file': val_split_file})
        train_dataset = MotionDatasetV2(**train_args)
        val_dataset = MotionDatasetV2(**val_args)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=1,
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=1,
                                shuffle=True, pin_memory=True)

    trainer.train(train_loader, val_loader, plot_t2m)
