from argparse import ArgumentParser
import argparse
from copy import deepcopy
import os
import json


def parse_and_load_from_model(parser, **kwargs):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    return load_from_model(args, parser, **kwargs)

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')

def load_from_model(args, parser, task=''):
    args_to_overwrite = []
    loaded_groups = ['dataset', 'model', 'diffusion']
    if task in ['multi_sample', 'multi_train']:
        loaded_groups.append('multi_person')
    elif task == 'inpainting' and args.inpainting_mask == '':
        loaded_groups.append('inpainting')
    
    for group_name in loaded_groups:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    args.model_path = args.model_path if task != 'multi_train' else args.pretrained_path
    # load args from model
    args_path = os.path.join(os.path.dirname(args.model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            args.__dict__[a] = model_args[a]
        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args

def parse_and_load_from_multiple_models(parser, task=''):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    model_paths = args.model_path.split(',')
    args_list = []
    for i in range(len(model_paths)):
        new_args = deepcopy(args)
        new_args.model_path = model_paths[i]
        args_list.append(load_from_model(new_args, parser, task))
    
    if task == 'inpainting' and args.inpainting_mask == '':
        inpainting_mask = ','.join([args.inpainting_mask for args in args_list])
        for args in args_list:
            args.inpainting_mask = inpainting_mask
    print(f'Using inpainting mask: {args_list[0].inpainting_mask}')
    return args_list


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--short_db", action='store_true', help="Load short babel for debug.")
    group.add_argument("--cropping_sampler", action='store_true', help="Load short babel for debug.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--use_tta", action='store_true', help="Time To Arrival position encoding")  # FIXME REMOVE?
    group.add_argument("--concat_trans_emb", action='store_true', help="Concat transition emb, else append after linear")  # FIXME REMOVE?
    group.add_argument("--trans_emb", action='store_true', help="Allow transition embedding")  # FIXME REMOVE?


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'amass', 'babel'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=200, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")

def add_multi_options(parser):
    group = parser.add_argument_group('multi_person')
    group.add_argument("--pretrained_path", default='./save/humanml_trans_enc_512/model000200000.pt', type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--multi_arch", default='trans_enc', type=str, choices=['trans_enc'], help="communication architecture.")
    group.add_argument("--multi_func", default='in_both_out_cur', type=str,
                       choices=['in_other_out_cur', 'in_both_out_cur', 'in_both_out_direct', 'in_both_out_cond'], help="Defines the IO of the communication block")
    group.add_argument("--multi_num_layers", default=2, type=int, help="communication num layers.")
    group.add_argument("--multi_backbone_split", default=8, type=int, help="Where to cut? 0 is at the start, num_layers at the end.")
    group.add_argument("--multi_mdm_freeze", default=1, type=int, help="If true freeze MDM.")
    group.add_argument("--multi_dataset", default='pw3d', choices=['pw3d'], type=str, help="Dataset name (choose from list).")
    group.add_argument("--multi_train_mode", default='prefix', choices=['prefix', 'text'], type=str, help="")
    group.add_argument("--multi_train_splits", default='train', type=str, help="coma separated")
    group.add_argument("--multi_eval_splits", default='validation', type=str, help="coma separated")
    parser.add_argument('--no_6dof', dest='predict_6dof', action='store_false')
    parser.set_defaults(predict_6dof=True)

def add_inpainting_options(parser):
    group = parser.add_argument_group('inpainting')
    group.add_argument("--inpainting_mask", default='', type=str, 
                       help="Comma separated list of masks to use. In sampling, if not specified, will load the mask from the used model/s. \
                       Every element could be one of: \
                           root, root_horizontal, in_between, prefix, upper_body, lower_body, \
                           or one of the joints in the humanml body format: \
                           pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2, left_ankle, right_ankle, spine3, left_foot, \
                            right_foot, neck, left_collar, right_collar, head, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,")
    group.add_argument("--no_filter_noise", action='store_false', dest='filter_noise',
                       help="When true, the noise will be filtered from the inpainted features.")
    parser.set_defaults(filter_noise=True)



def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=1, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    # group.add_argument("--ar_steps", default=3, type=int,
    #                    help="Number of auto-regressive steps.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")


def add_double_take_options(parser):
    group = parser.add_argument_group('double_take')
    # group.add_argument("--double_take", action='store_true',
    #                    help="double take on the generated motion")
    group.add_argument("--double_take", default=True, type=bool,
                       help="double take on the generated motion")
    group.add_argument("--second_take_only", action='store_true',
                       help="double take on the generated motion")
    group.add_argument("--handshake_size", default=20, type=int,
                       help="handshake size for unfolding")
    group.add_argument("--blend_len", default=20, type=int,
                       help="blending with linear mask length")
    group.add_argument("--repaint_rep", default=10, type=int,
                       help="number of times to sample during repaint")
    group.add_argument("--repaint", action='store_true',
                       help="use repaint")
    group.add_argument("--debug_double_take", action='store_true',
                       help="double_take debug mode")
    group.add_argument("--skip_steps_double_take", default=100, type=int,
                       help="number of times to sample during repaint")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--sample_gt", action='store_true',
                       help="sample and visualize gt instead of generate sample")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', type=str,
                       help="Defines which parts of the input motion will be edited.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_edit_inpainting_options(parser):
    add_inpainting_options(parser)
    group = parser.add_argument_group('edit')
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--show_input", action='store_true',
                       help="If true, will show the motion from which the inpainting features were taken.")

def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--num_unfoldings", default=1, type=int,
                       help="How many unfolding sequences to use. 1 is evaluation withou unfolding.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--transition_margins", default=0, type=int,
                       help="For evaluation - take margin around transition")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will delete the existing lof file (if it has the same name).")
    group.add_argument("--replication_times", default=None, type=int,
                       help="Number of evaluation iterations to apply.")


def add_evaluation_double_take_options(parser):
    group = parser.add_argument_group('eval_double_take')
    group.add_argument("--eval_on", default='motion', type=str, choices=['motion', 'transition'],
                    help="For evaluation - choose to eval over motion or transition")


def add_frame_sampler_options(parser):
    group = parser.add_argument_group('framesampler')
    group.add_argument("--min_seq_len", default=45, type=int,
                       help="babel dataset FrameSampler minimum length")
    group.add_argument("--max_seq_len", default=250, type=int,
                       help="babel dataset FrameSampler maximum length")


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_frame_sampler_options(parser)
    return parser.parse_args()

def train_multi_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_multi_options(parser)
    add_training_options(parser)
    return parse_and_load_from_model(parser, task='multi_train')

def train_inpainting_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_inpainting_options(parser)
    return parser.parse_args()

def generate_multi_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_multi_options(parser)
    add_generate_options(parser)
    return parse_and_load_from_model(parser, task='multi_sample')

def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_frame_sampler_options(parser)
    add_double_take_options(parser)
    return parse_and_load_from_model(parser)


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def edit_inpainting_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_inpainting_options(parser)
    return parse_and_load_from_multiple_models(parser, task='inpainting')


def edit_double_take_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    add_frame_sampler_options(parser)
    add_double_take_options(parser)
    return parse_and_load_from_model(parser)


def edit_multi_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_multi_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser, task='multi_sample')


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_double_take_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    add_frame_sampler_options(parser)
    add_double_take_options(parser)
    add_evaluation_double_take_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_multi_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_multi_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser, task='multi_sample')

def evaluation_inpainting_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    add_inpainting_options(parser)
    return parse_and_load_from_multiple_models(parser, task='inpainting')

def smplh_args():
    parser = ArgumentParser()
    add_base_options(parser)
    return parse_and_load_from_model(parser)
