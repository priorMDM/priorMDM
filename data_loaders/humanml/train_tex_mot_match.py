import os

from os.path import join as pjoin
import torch

from data_loaders.amass.sampling import FrameSampler
from data_loaders.amass.transforms import SMPLTransform
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.options.train_options import TrainTexMotMatchOptions

from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import TextMotionMatchTrainer
from data_loaders.humanml.data.dataset import Text2MotionDatasetV2, collate_fn, BABEL_Text2MotionDatasetV2
from data_loaders.humanml.scripts.motion_process import *
from torch.utils.data import DataLoader
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer, POS_enumerator
from copy import deepcopy
from  data_loaders.humanml import collect_babel_stats


def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose - opt.foot_contact_entries, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=dim_word,
                                  pos_size=dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)
    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)
    if not opt.is_continue and not opt.movement_from_scratch:
       checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
                               map_location=opt.device)
       movement_enc.load_state_dict(checkpoint['movement_enc'])
    return text_enc, motion_enc, movement_enc


if __name__ == '__main__':
    parser = TrainTexMotMatchOptions()
    opt = parser.parse()

    opt.device = torch.device("cpu" if opt.gpu_id==-1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)
    if opt.gpu_id != -1:
        # self.opt.gpu_id = int(self.opt.gpu_id)
        torch.cuda.set_device(opt.gpu_id)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)
    opt.eval_dir = pjoin(opt.save_root, 'eval')

    os.makedirs(opt.model_dir, exist_ok=True)
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
        num_classes = 200 // opt.unit_length
        meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'Comp_v6_KLD01', 'meta')
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        opt.dim_pose = 251
        opt.max_motion_length = 196
        num_classes = 200 // opt.unit_length
        meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'Comp_v6_KLD005', 'meta')
    elif opt.dataset_name == 'babel':
        opt.foot_contact_entries = 0
        opt.dim_pose = 135
        opt.is_continue = False
        opt.min_motion_length = 15
        opt.max_motion_length = 480
        opt.max_epoch //= 4
        # meta_root = pjoin(opt.checkpoints_dir, opt.dataset_name, 'motion1', 'meta')
        # if not os.path.exists(meta_root):
        #     collect_babel_stats.run()
    else:
        raise KeyError('Dataset Does Not Exist')

    opt.dataset = opt.dataset_name  # For clearml
    dim_word = 300
    dim_pos_ohot = len(POS_enumerator)


    text_encoder, motion_encoder, movement_encoder = build_models(opt)

    pc_text_enc = sum(param.numel() for param in text_encoder.parameters())
    print(text_encoder)
    print("Total parameters of text encoder: {}M".format(pc_text_enc//1e6))
    pc_motion_enc = sum(param.numel() for param in motion_encoder.parameters())
    print(motion_encoder)
    print("Total parameters of motion encoder: {}M".format(pc_motion_enc//1e6))
    print("Total parameters: {}M".format((pc_motion_enc + pc_text_enc)//1e6))


    trainer = TextMotionMatchTrainer(opt, text_encoder, motion_encoder, movement_encoder)

    if opt.dataset_name == 'babel':
        train_loader = get_dataset_loader('babel',
                                          batch_size=opt.batch_size, num_frames=480,  # not in use
                                          split='train', load_mode='evaluator_train', opt=opt)
        val_loader = get_dataset_loader('babel',
                                        batch_size=opt.batch_size, num_frames=480,  # not in use
                                        split='val', load_mode='evaluator_train', opt=opt)

    else:
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        mean = np.load(pjoin(meta_root, 'mean.npy'))
        std = np.load(pjoin(meta_root, 'std.npy'))
        dataset_args = {
            'opt': opt, 'mean': mean, 'std': std, 'w_vectorizer': w_vectorizer
        }
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
        val_args, train_args = deepcopy(dataset_args), deepcopy(dataset_args)
        train_args.update({'split_file': train_split_file})
        val_args.update({'split_file': val_split_file})

        train_dataset = Text2MotionDatasetV2(**train_args)
        val_dataset = Text2MotionDatasetV2(**val_args)
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                                  shuffle=True, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                                shuffle=True, collate_fn=collate_fn, pin_memory=True)  # FIXME
        # val_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
        #                         shuffle=True, collate_fn=collate_fn, pin_memory=True)

    trainer.train(train_loader, val_loader)