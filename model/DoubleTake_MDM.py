import torch
import torch.nn as nn
from model.rotation2xyz import Rotation2xyz

from model.mdm import MDM
from model.mdm import InputProcess
from model.mdm import PositionalEncoding

class doubleTake_MDM(MDM):
    # def __init__(self, **kargs):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):

        super(doubleTake_MDM, self).__init__(modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         arch, emb_trans_dec, clip_version, **kargs)

        assert self.arch == 'trans_enc' # we evaluate only on trans_enc arch

        self.use_tta = False #use_tta # TODO: remove as we don't show tta feature
        self.trans_emb = kargs.get('trans_emb', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)  # mask all to remove condition
        trans_emb_dim = 0
        if self.trans_emb:
            print("Using TransEmb!")
            trans_emb_dim = 4
            self.emb_trans_or_not = nn.Embedding(2, trans_emb_dim)

        self.input_process = InputProcess(self.data_rep, (self.input_feats + trans_emb_dim)
                                                if self.trans_emb else self.input_feats, self.latent_dim)
        # self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, use_tta=self.use_tta,
        #                                                     max_len=5000)
        #
        # self.t_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, use_tta=False, max_len=5000)
        # self.embed_timestep = self.TimestepEmbedder(self.latent_dim, self.t_pos_encoder if self.use_tta else
        #                                             self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'action' in self.cond_mode:
                raise Exception("cond_mode action not implemented yet")

        # self.output_process = self.OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
        #                                         self.nfeats)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset, batch_size=kargs.get('batch_size', None))


    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape

        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        if self.trans_emb:
            transitions = y.get('is_transition', False)
            transitions = transitions.to(x.device)
            transitions_emb = self.emb_trans_or_not(transitions.long()) # [bs, seq_len, dim]
            transitions_emb = transitions_emb.permute(0, 2, 1).unsqueeze(2)
            x = torch.cat((x, transitions_emb), dim=1)

        x = self.input_process(x) #[seqlen, bs, d]

        if self.arch == 'trans_enc':
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)#, y['lengths'])  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # [seqlen, bs, d]
        else:
            raise Exception("Only trained for transformer Encoder")

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output