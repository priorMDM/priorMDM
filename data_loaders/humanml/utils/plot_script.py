import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from data_loaders import humanml_utils


MAX_LINE_LENGTH = 20


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[], handshake_size=0, blend_size=0, step_sizes=[], lengths = [], joints2=None, painting_features=[]):
    matplotlib.use('Agg')
    """
    A wrapper around explicit_plot_3d_motion that 
    uses gt_frames to determine the colors of the frames
    """
    data = joints.copy().reshape(len(joints), -1, 3)
    frames_number = data.shape[0]
    frame_colors = ['blue' if index in gt_frames else 'orange' for index in range(frames_number)]
    if vis_mode == 'unfold':
        frame_colors = ['purple'] *handshake_size + ['blue']*blend_size + ['orange'] *(120-handshake_size*2-blend_size*2) +['orange']*blend_size
        frame_colors = ['orange'] *(120-handshake_size-blend_size) + ['orange']*blend_size + frame_colors*1024
    elif vis_mode == 'unfold_arb_len':
        for ii, step_size in enumerate(step_sizes):
            if ii == 0:
                frame_colors = ['orange']*(step_size - handshake_size - blend_size) + ['orange']*blend_size + ['purple'] * (handshake_size//2)
                continue
            if ii == len(step_sizes)-1:
                frame_colors += ['purple'] * (handshake_size//2) + ['orange'] * blend_size + ['orange'] * (lengths[ii] - handshake_size - blend_size)
                continue
            frame_colors += ['purple'] * (handshake_size // 2) + ['orange'] * blend_size + ['orange'] * (
                            lengths[ii] - 2 * handshake_size - 2 * blend_size) + ['orange'] * blend_size + \
                            ['purple'] * (handshake_size // 2)
    elif vis_mode == 'gt':
        frame_colors = ['blue'] * frames_number
    explicit_plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=figsize, fps=fps, radius=radius, vis_mode=vis_mode, frame_colors=frame_colors, joints2=joints2, painting_features=painting_features)



def explicit_plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3, vis_mode="default", frame_colors=[], joints2=None, painting_features=[]):
    """
    outputs the 3D motion to an mp4 file
    """
    matplotlib.use("Agg")

    if type(title) == str:
        title = ["\n".join(wrap(title, 20))]
    elif type(title) == list:
        title = ["\n".join(wrap(t, 20)) for t in title]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title[0], fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    data2 = None
    if joints2 is not None:
        data2 = joints2.copy().reshape(len(joints), -1, 3)

    # preparation related to specific datasets
    if dataset == "kit":
        data *= 0.003  # scale for visualization
    elif dataset == "humanml":
        data *= 1.3  # scale for visualization
        if data2 is not None:
            data2 *= 1.3
    elif dataset in ["humanact12", "uestc"]:
        data *= -1.5  # reverse axes, scale for visualization
    elif dataset in ['humanact12', 'uestc', 'amass']:
        data *= -1.5 # reverse axes, scale for visualization
    elif dataset =='babel':
        data *= 1.3


    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    if data2 is not None:
        MINS = np.concatenate((data, data2)).min(axis=0).min(axis=0)
        MAXS = np.concatenate((data, data2)).max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors_purple = ["#6B31DB", "#AD40A8", "#AF2B79", "#9B00FF", "#D836C1"]

    colors_upper_body = colors_blue[:2] + colors_orange[2:]

    colors_dict = {"blue": colors_blue, "orange": colors_orange, "purple": colors_purple, "upper_body": colors_upper_body}

    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0].copy()

    # Reduce data2 first before overriding root position with zeros
    if data2 is not None:
        data2[:, :, 1] -= height_offset
        data2[..., 0] -= data[:, 0:1, 0]
        data2[..., 2] -= data[:, 0:1, 2]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        if len(title) > 1:
            fig.suptitle(title[index], fontsize=10)
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 2], MAXS[2] - trajec[index, 2])


        used_colors = colors_dict[frame_colors[index]] if (index < len(frame_colors)) else colors_dict["blue"]
        other_colors = used_colors  # colors_purple
        for i, (chain, color, other_color) in enumerate(zip(kinematic_tree, used_colors, other_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
            if data2 is not None:
                ax.plot3D(data2[index, chain, 0], data2[index, chain, 1], data2[index, chain, 2], linewidth=linewidth, color=other_color)
        
        def plot_root_horizontal():
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 1]), trajec[:index, 2] - trajec[index, 2], linewidth=2.0,
                      color=used_colors[0])
        
        def plot_root():
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], trajec[:index, 1], trajec[:index, 2] - trajec[index, 2], linewidth=2.0,
                      color=used_colors[0])
        
        def plot_feature(feature):
            # trajectory = Line3DCollection(joints[:,0])
            if feature in humanml_utils.HML_JOINT_NAMES:
                feat_index = humanml_utils.HML_JOINT_NAMES.index(feature)
                ax.plot3D(data[:index+1, feat_index, 0] + (trajec[:index+1, 0] - trajec[index, 0]),
                          data[:index+1, feat_index, 1],
                          data[:index+1, feat_index, 2] + (trajec[:index+1, 2] - trajec[index, 2]), linewidth=2.0,
                        color=used_colors[0])
        
        if 'root_horizontal' in painting_features:
            plot_root_horizontal()
        if 'root' in painting_features:
            plot_root()
        for feat in painting_features:
            plot_feature(feat)
            
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 // fps, repeat=False)

    ani.save(save_path, fps=fps)

    plt.close()
