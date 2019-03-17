import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from typing import Optional, Tuple

from lib.tree_lstm import tree_utils
from lib.tree_lstm.tree_lstm import BiTreeLSTM_Backward, BiTreeLSTM_Foreward
from lib.fpn.box_utils import nms_overlaps
from lib.word_vectors import obj_edge_vectors
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
import numpy as np

class DecoderTreeLSTM(torch.nn.Module):
    def __init__(self, classes, embed_dim, inputs_dim, hidden_dim, direction='backward', dropout=0.2, pass_root = False, not_rl = True):
        super(DecoderTreeLSTM, self).__init__()
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention purposes
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_size: Number of words in the vocab
        :param bos_token: To use during decoding (non teacher forcing mode))
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        direction = foreward | backward
        """
        self.classes = classes
        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.nms_thresh = 0.5
        self.dropout = dropout
        self.pass_root = pass_root
        # generate embed layer
        embed_vecs = obj_edge_vectors(['start'] + self.classes, wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.classes), embed_dim)
        self.obj_embed.weight.data = embed_vecs
        # generate out layer
        self.out = nn.Linear(self.hidden_size, len(self.classes))
        self.out.weight = torch.nn.init.xavier_normal(self.out.weight,  gain=1.0)
        self.out.bias.data.fill_(0.0)
        self.not_rl = not_rl

        if direction == 'backward':
            self.input_size = inputs_dim + embed_dim
            self.decoderLSTM = BiTreeLSTM_Backward(self.input_size, self.hidden_size, self.pass_root, is_pass_embed=True, embed_layer=self.obj_embed, embed_out_layer=self.out, not_rl=not_rl)
        elif direction == 'foreward':
            self.input_size = inputs_dim + embed_dim * 2
            self.decoderLSTM = BiTreeLSTM_Foreward(self.input_size, self.hidden_size, self.pass_root, is_pass_embed=True, embed_layer=self.obj_embed, embed_out_layer=self.out, not_rl=not_rl)
        else:
            print('Error Decoder LSTM Direction')


    def forward(self, forest, features, num_obj, labels=None, boxes_for_nms=None, batch_size=0):
        # generate dropout
        if self.dropout > 0.0:
            dropout_mask = get_dropout_mask(self.dropout, self.hidden_size)
        else:
            dropout_mask = None

        # generate tree lstm input/output class
        out_h = None
        out_dists = None
        out_commitments = None
        h_order = Variable(torch.LongTensor(num_obj).zero_().cuda())
        order_idx = 0
        lstm_io = tree_utils.TreeLSTM_IO(out_h, h_order, order_idx, out_dists, out_commitments, dropout_mask)

        for idx in range(len(forest)):
            self.decoderLSTM(forest[idx], features, lstm_io)

        out_h = torch.index_select(lstm_io.hidden, 0, lstm_io.order.long())
        out_dists = torch.index_select(lstm_io.dists, 0, lstm_io.order.long())[:-batch_size]
        out_commitments = torch.index_select(lstm_io.commitments, 0, lstm_io.order.long())[:-batch_size]

        # Do NMS here as a post-processing step
        if boxes_for_nms is not None and not self.training and self.not_rl:
            is_overlap = nms_overlaps(boxes_for_nms.data).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh
            # is_overlap[np.arange(boxes_for_nms.size(0)), np.arange(boxes_for_nms.size(0))] = False

            out_dists_sampled = F.softmax(out_dists, 1).data.cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = out_commitments.data.new(out_commitments.shape[0]).fill_(0)

            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            out_commitments = Variable(out_commitments.view(-1))
        else:
            out_commitments = out_commitments.view(-1)

        if self.training and self.not_rl and (labels is not None):
            out_commitments = labels.clone()
        else:
            out_commitments = torch.cat((out_commitments, Variable(torch.randn(batch_size).long().fill_(0).cuda()).view(-1)), 0)
            
        return out_dists, out_commitments


def get_dropout_mask(dropout_probability: float, h_dim: int):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.

    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.


    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = Variable(torch.FloatTensor(h_dim).cuda().fill_(0.0))
    binary_mask.data.copy_(torch.rand(h_dim) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask