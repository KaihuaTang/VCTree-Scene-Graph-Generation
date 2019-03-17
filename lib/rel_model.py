"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

# from lib.decoder_rnn import DecoderRNN, lstm_factory, LockedDropout
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import math
from config import IM_SCALE, ROOT_PATH, CO_OCCOUR_PATH

# import tree lstm
from lib.tree_lstm import tree_lstm, gen_tree, tree_utils
from lib.tree_lstm.draw_tree import draw_tree_region, draw_tree_region_v2
from lib.tree_lstm.decoder_tree_lstm import DecoderTreeLSTM
from lib.tree_lstm.graph_to_tree import graph_to_trees, arbitraryForest_to_biForest


def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds[-1] + 1
    rois_per_image = scores.new(num_im)
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())

    # ~~~~~~~~~~~~~~~~
    # HACKY CODE ALERT!!!
    # we're sorting by confidence which is in the range (0,1), but more importantly by longest
    # img....
    # ~~~~~~~~~~~~~~~~
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed

MODES = ('sgdet', 'sgcls', 'predcls')


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """
    def __init__(self, classes, rel_classes, mode='sgdet',
                 embed_dim=200, hidden_dim=256, obj_dim=2048,
                 nl_obj=2, nl_edge=2, dropout_rate=0.2, order='confidence',
                 pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True,
                 use_rl_tree=True, draw_tree=False):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.nl_obj = nl_obj
        self.nl_edge = nl_edge

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.dropout_rate = dropout_rate
        self.pass_in_obj_feats_to_decoder = pass_in_obj_feats_to_decoder
        self.pass_in_obj_feats_to_edge = pass_in_obj_feats_to_edge
        self.use_rl_tree = use_rl_tree
        self.draw_tree = draw_tree

        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order

        # EMBEDDINGS
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()
        self.virtual_node_embed = nn.Embedding(1, self.embed_dim) # used to encode Root Node

        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),
        ])

        # generate tree
        self.rl_input_size = 256
        self.rl_hidden_size = 256
        self.feat_preprocess_net = gen_tree.RLFeatPreprocessNet(self.obj_dim, self.embed_dim, 8, 6, self.rl_input_size)
        self.rl_sub = nn.Linear(self.rl_input_size, self.rl_hidden_size)
        self.rl_obj = nn.Linear(self.rl_input_size, self.rl_hidden_size)
        self.rl_scores = nn.Linear(self.rl_hidden_size * 3 + 3, 1)  # (left child score, right child score)
        # init
        self.rl_sub.weight = torch.nn.init.xavier_normal(self.rl_sub.weight,  gain=1.0)
        self.rl_sub.bias.data.zero_()
        self.rl_obj.weight = torch.nn.init.xavier_normal(self.rl_obj.weight,  gain=1.0)
        self.rl_obj.bias.data.zero_()
        self.rl_scores.weight = torch.nn.init.xavier_normal(self.rl_scores.weight,  gain=1.0)
        self.rl_scores.bias.data.zero_()

        # whether draw tree
        if self.draw_tree:
            self.draw_tree_count = 0
            self.draw_tree_max = 600

        if self.nl_obj > 0:
            self.obj_tree_lstm = tree_lstm.MultiLayer_BTreeLSTM(self.obj_dim+self.embed_dim+128, self.hidden_dim, self.nl_obj, dropout_rate)

            decoder_inputs_dim = self.hidden_dim
            if self.pass_in_obj_feats_to_decoder:
                decoder_inputs_dim += self.obj_dim + self.embed_dim

            self.decoder_tree_lstm = DecoderTreeLSTM(classes, embed_dim=100, #embed_dim = self.embed_dim, 
                                          inputs_dim=decoder_inputs_dim, 
                                          hidden_dim=self.hidden_dim, 
                                          direction = 'backward',
                                          dropout=dropout_rate,
                                          pass_root=False,
                                          not_rl = not self.use_rl_tree)
        else:
            self.decoder_lin = nn.Linear(self.obj_dim + self.embed_dim + 128, self.num_classes)

        if self.nl_edge > 0:
            input_dim = self.embed_dim
            if self.nl_obj > 0:
                input_dim += self.hidden_dim
            if self.pass_in_obj_feats_to_edge:
                input_dim += self.obj_dim

            self.edge_tree_lstm = tree_lstm.MultiLayer_BTreeLSTM(input_dim, self.hidden_dim, self.nl_edge, dropout_rate)


    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:,2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
        elif self.order == 'leftright':
            centers = cxcywh[:,0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return _sort_by_score(batch_idx, scores)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def edge_ctx(self, obj_feats, obj_preds, box_priors=None, forest = None):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """

        # Only use hard embeddings
        obj_embed2 = self.obj_embed2(obj_preds)
        inp_feats = torch.cat((obj_embed2, obj_feats), 1)
        # use bidirectional tree lstm to update
        edge_ctx = self.edge_tree_lstm(forest, inp_feats, box_priors.shape[0])
        return edge_ctx

    def obj_ctx(self, obj_feats, obj_labels=None, box_priors=None, boxes_per_cls=None, forest = None, batch_size=0):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # use bidirectional tree lstm to update
        encoder_rep = self.obj_tree_lstm(forest, obj_feats, box_priors.shape[0])

        # Decode in order
        if self.mode != 'predcls':
            decode_feature = torch.cat((obj_feats, encoder_rep), 1) if self.pass_in_obj_feats_to_decoder else encoder_rep
            obj_dists, obj_preds = self.decoder_tree_lstm(forest, decode_feature, 
                                         box_priors.shape[0], 
                                         labels=obj_labels if obj_labels is not None else None, 
                                         boxes_for_nms=boxes_per_cls if boxes_per_cls is not None else None,
                                         batch_size=batch_size)
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = Variable(to_onehot(obj_preds.data[:-batch_size], self.num_classes))

        return obj_dists, obj_preds, encoder_rep

    def rl_score_net(self, prepro_feat, obj_distributions, co_occour, rel_labels, batch_size, im_inds, pseudo_im_inds):
        """
        obj_distributions [num_obj, 150]
        """
        co_occour_var = Variable(torch.from_numpy(co_occour).float().cuda())
        label_scores = co_occour_var.sum(1).view(-1)
        # generate RL score
        rl_sub_feat = self.rl_sub(prepro_feat)
        rl_obj_feat = self.rl_obj(prepro_feat)
        rl_sub_feat = F.relu(rl_sub_feat)
        rl_obj_feat = F.relu(rl_obj_feat)
        # obj_num = real num of obj + num of batch size
        obj_num = rl_sub_feat.shape[0]
        hidden_size = rl_sub_feat.shape[1]
        num_class = obj_distributions.shape[1] # 150

        rl_sub_feat = rl_sub_feat.view(1, obj_num, hidden_size).expand(obj_num, obj_num, hidden_size)
        rl_obj_feat = rl_obj_feat.view(obj_num, 1, hidden_size).expand(obj_num, obj_num, hidden_size)
        sub_dist = obj_distributions.view(1, obj_num, num_class).expand(obj_num, obj_num, num_class).contiguous().view(-1, num_class)
        obj_dist = obj_distributions.view(obj_num, 1, num_class).expand(obj_num, obj_num, num_class).contiguous().view(-1, num_class)
        mat_dot_dis = sub_dist.view(-1, num_class, 1) @ obj_dist.view(-1, 1, num_class) # [num_pair, 150, 150]
        ele_dot_score = mat_dot_dis * co_occour_var # [num_pair, 150, 150]

        pair_score = ele_dot_score.view(obj_num*obj_num, num_class*num_class).sum(1).view(obj_num, obj_num, 1).detach()
        sub_score = (sub_dist * label_scores).sum(1).view(obj_num, obj_num, 1).detach()
        obj_score = (obj_dist * label_scores).sum(1).view(obj_num, obj_num, 1).detach()

        # pair_score
        pair_feat = torch.cat((rl_sub_feat * rl_obj_feat, rl_sub_feat, rl_obj_feat, pair_score, sub_score, obj_score), 2)
        #assert(pair_feat.shape[2] == hidden_size * 3)
        pair_output = self.rl_scores(pair_feat.view(-1, hidden_size * 3 + 3))
        pair_gates = F.sigmoid(pair_output).view(-1,1) # (relation prob)

        if self.mode == 'sgdet':
            im_inds_sub = pseudo_im_inds.view(1, obj_num).expand(obj_num, obj_num)
            im_inds_obj = pseudo_im_inds.view(obj_num, 1).expand(obj_num, obj_num)
            im_inds_mask = (im_inds_sub == im_inds_obj).float().view(-1,1).detach()
            pair_gates = pair_gates * im_inds_mask

        # add pair score offset
        relation_scores = (pair_score.view(-1,1) * pair_gates).view(obj_num, obj_num) 
        # generate ground truth relation label 0 or 1 for all obj pairs
        if self.training and not self.use_rl_tree:
            gt_rel_label = Variable(torch.FloatTensor(pair_gates.shape[0]).zero_().cuda()).view(obj_num, obj_num)

            for i in range(rel_labels.shape[0]):
                if int(rel_labels[i, 3]) != 0:
                    sub_id = int(rel_labels[i, 1])
                    obj_id = int(rel_labels[i, 2])
                    gt_rel_label[sub_id, obj_id] = 1
                    gt_rel_label[obj_id, sub_id] = 1
            for i in range(batch_size):  # virtual node connect to every node
                related_node_index = torch.nonzero(im_inds==i)  # True im_inds, not pseudo im_inds
                start_index = int(related_node_index[0])
                end_index = int(related_node_index[-1]) + 1
                gt_rel_label[-batch_size+i, start_index:end_index] = 1
                gt_rel_label[start_index:end_index, -batch_size+i] = 1

        else:
            gt_rel_label = None

        return relation_scores, pair_gates.view(-1), gt_rel_label


    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None, gt_forest=None, image_rois=None, image_fmap=None, co_occour=None, rel_labels=None, origin_img=None):
        """
        Forward pass through the object and edge context
        :param obj_priors: [obj_num, (x1,y1,x2,y2)], float cuda
        :param obj_fmaps:
        :param im_inds: [obj_num] long variable
        :param obj_labels:
        :param boxes:
        :return:
        """
        if self.mode == 'predcls':
            obj_logits = Variable(to_onehot(obj_labels.data, self.num_classes))
            
        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight
        
        batch_size = image_rois.shape[0]
        # pseudo box and image index: to encode virtual node into original inputs
        pseudo_box_priors = torch.cat((box_priors, image_rois[:, 1:].contiguous().data), 0)  # [obj_num + batch_size, 4]
        pseudo_im_inds = torch.cat((im_inds, image_rois[:,0].contiguous().long().view(-1)), 0) # [obj_num + batch_size]
        pseudo_obj_fmaps = torch.cat((obj_fmaps.clone().detach(), image_fmap.detach()), 0)  # [obj_num + batch_size, 4096]
        virtual_embed = self.virtual_node_embed.weight[0].view(1, -1).expand(batch_size, -1)
        pseudo_obj_embed = torch.cat((obj_embed, virtual_embed), 0) # [obj_num + batch_size, embed_dim]
        if self.training or (self.mode == 'predcls'):
            pseudo_obj_labels = torch.cat((obj_labels, Variable(torch.randn(1).fill_(0).cuda()).expand(batch_size).long().view(-1)), 0)
        else:
            pseudo_obj_labels = None
        
        if self.mode == 'sgdet':
            obj_distributions = F.softmax(obj_logits, dim=1)[:,1:]
        else:
            obj_distributions = F.softmax(obj_logits[:,1:], dim=1)
        pseudo_obj_distributions = torch.cat((obj_distributions, Variable(torch.randn(batch_size, obj_distributions.shape[1]).fill_(0).cuda())), 0)
        # generate RL gen tree input
        box_embed = tree_utils.get_box_info(Variable(pseudo_box_priors)) # 8-digits
        overlap_embed, _ = tree_utils.get_overlap_info(pseudo_im_inds, Variable(pseudo_box_priors)) # 4-digits
        prepro_feat = self.feat_preprocess_net(pseudo_obj_fmaps, pseudo_obj_embed, box_embed, overlap_embed)
        pair_scores, pair_rel_gate, pair_rel_gt = self.rl_score_net(prepro_feat, pseudo_obj_distributions, co_occour, rel_labels, batch_size, im_inds, pseudo_im_inds)

        #print('node_scores', node_scores.data.cpu().numpy())
        arbitrary_forest, gen_tree_loss, entropy_loss = gen_tree.generate_forest(pseudo_im_inds, gt_forest, pair_scores, Variable(pseudo_box_priors), pseudo_obj_labels, self.use_rl_tree, self.training, self.mode)
        forest = arbitraryForest_to_biForest(arbitrary_forest)

        pseudo_pos_embed = self.pos_embed(Variable(center_size(pseudo_box_priors)))
        obj_pre_rep = torch.cat((pseudo_obj_fmaps, pseudo_obj_embed, pseudo_pos_embed), 1)
        if self.nl_obj > 0:
            obj_dists2, obj_preds, obj_ctx = self.obj_ctx(
                obj_pre_rep,
                pseudo_obj_labels,
                pseudo_box_priors,
                boxes_per_cls,
                forest,
                batch_size
            )
        else:
            print('Error, No obj ctx')

        edge_ctx = None
        if self.nl_edge > 0:
            edge_ctx = self.edge_ctx(
                torch.cat((pseudo_obj_fmaps, obj_ctx), 1) if self.pass_in_obj_feats_to_edge else obj_ctx,
                obj_preds=obj_preds,
                box_priors=pseudo_box_priors,
                forest = forest,
            )

        # draw tree
        if self.draw_tree and (self.draw_tree_count < self.draw_tree_max):
            for tree_idx in range(len(forest)):
                draw_tree_region(forest[tree_idx], origin_img, self.draw_tree_count)
                draw_tree_region_v2(forest[tree_idx], origin_img, self.draw_tree_count, obj_preds)
                self.draw_tree_count += 1

        # remove virtual nodes
        return obj_dists2, obj_preds[:-batch_size], edge_ctx[:-batch_size], gen_tree_loss, entropy_loss, pair_rel_gate, pair_rel_gt


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """
    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=1, nl_edge=2, use_resnet=False, order='confidence', thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True, rec_dropout=0.1, use_bias=True, use_tanh=True, use_encoded_box=True, use_rl_tree=True, draw_tree=False,
                 limit_vision=True):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode
        self.co_occour = np.load(CO_OCCOUR_PATH)
        self.co_occour = self.co_occour / self.co_occour.sum()

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim

        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.use_encoded_box = use_encoded_box
        self.use_rl_tree = use_rl_tree
        self.draw_tree = draw_tree
        self.limit_vision=limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.rl_train = False

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
            use_rl_tree = self.use_rl_tree
        )

        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                                         obj_dim=self.obj_dim,
                                         nl_obj=nl_obj, nl_edge=nl_edge, dropout_rate=rec_dropout,
                                         order=order,
                                         pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
                                         pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge,
                                         use_rl_tree=self.use_rl_tree,
                                         draw_tree = self.draw_tree)

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(use_dropout=False, pretrained=False).classifier


        ###################################
        self.post_lstm = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.

        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
        self.post_lstm.bias.data.zero_()
        self.post_cat.weight = torch.nn.init.xavier_normal(self.post_cat.weight,  gain=1.0)
        self.post_cat.bias.data.zero_()

        if self.use_encoded_box:
            # encode spatial info
            self.encode_spatial_1 = nn.Linear(32, 512)
            self.encode_spatial_2 = nn.Linear(512, self.pooling_dim)

            self.encode_spatial_1.weight.data.normal_(0, 1.0)
            self.encode_spatial_1.bias.data.zero_()
            self.encode_spatial_2.weight.data.normal_(0, 0.1)
            self.encode_spatial_2.bias.data.zero_()

        if nl_edge == 0:
            self.post_emb = nn.Embedding(self.num_classes, self.pooling_dim*2)
            self.post_emb.weight.data.normal_(0, math.sqrt(1.0))

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias()

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training and not self.use_rl_tree:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def get_rel_label(self, im_inds, gt_rels, rel_inds):
        np_im_inds = im_inds.data.cpu().numpy()
        np_gt_rels = gt_rels.long().data.cpu().numpy()
        np_rel_inds = rel_inds.long().cpu().numpy()

        num_obj = int(im_inds.shape[0])
        sub_id = np_rel_inds[:, 1]
        obj_id = np_rel_inds[:, 2]
        select_id = sub_id * num_obj + obj_id

        count = 0
        offset = 0
        slicedInds = np.where(np_im_inds == count)[0]

        label = np.array([0]*num_obj*num_obj, dtype=int)
        while(len(slicedInds) > 0):
            slice_len = len(slicedInds)
            selectInds = np.where(np_gt_rels[:,0] == count)[0]
            slicedRels = np_gt_rels[selectInds,:]
            flattenID = (slicedRels[:,1] + offset) * num_obj + (slicedRels[:,2] + offset)
            slicedLabel = slicedRels[:,3]

            label[flattenID] = slicedLabel
            
            count += 1
            offset += slice_len
            slicedInds = np.where(np_im_inds == count)[0]
        
        return Variable(torch.from_numpy(label[select_id]).long().cuda())


    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels, fg_rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        #if self.training and (not self.use_rl_tree):
            # generate arbitrary forest according to graph
        #    arbitrary_forest = graph_to_trees(self.co_occour, result.rel_labels, gt_classes)
        #else:
        arbitrary_forest = None

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)

        if self.use_rl_tree:
            result.rel_label_tkh = self.get_rel_label(im_inds, gt_rels, rel_inds)

        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        # whole image feature, used for virtual node
        batch_size = result.fmap.shape[0]
        image_rois = Variable(torch.randn(batch_size, 5).fill_(0).cuda())
        for i in range(batch_size):
            image_rois[i, 0] = i
            image_rois[i, 1] = 0
            image_rois[i, 2] = 0
            image_rois[i, 3] = IM_SCALE
            image_rois[i, 4] = IM_SCALE
        image_fmap = self.obj_feature_map(result.fmap.detach(), image_rois)

        if self.mode != 'sgdet' and self.training:
            fg_rel_labels = result.rel_labels

        # Prevent gradients from flowing back into score_fc from elsewhere
        result.rm_obj_dists, result.obj_preds, edge_ctx, result.gen_tree_loss, result.entropy_loss, result.pair_gate, result.pair_gt = self.context(
            result.obj_fmap,
            result.rm_obj_dists.detach(),
            im_inds, result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
            boxes.data, result.boxes_all, 
            arbitrary_forest,
            image_rois,
            image_fmap,
            self.co_occour,
            fg_rel_labels if self.training else None,
            x)

        if edge_ctx is None:
            edge_rep = self.post_emb(result.obj_preds)
        else:
            edge_rep = self.post_lstm(edge_ctx)

        # Split into subject and object representations
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)

        subj_rep = edge_rep[:, 0]
        obj_rep = edge_rep[:, 1]

        prod_rep =  torch.cat((subj_rep[rel_inds[:, 1]], obj_rep[rel_inds[:, 2]]), 1)
        prod_rep = self.post_cat(prod_rep)

        if self.use_encoded_box:
            # encode spatial info
            assert(boxes.shape[1] == 4)
            # encoded_boxes: [box_num, (x1,y1,x2,y2,cx,cy,w,h)]
            encoded_boxes = tree_utils.get_box_info(boxes)
            # encoded_boxes_pair: [batch_szie, (box1, box2, unionbox, intersectionbox)]
            encoded_boxes_pair = tree_utils.get_box_pair_info(encoded_boxes[rel_inds[:, 1]], encoded_boxes[rel_inds[:, 2]])
            # encoded_spatial_rep
            spatial_rep = F.relu(self.encode_spatial_2(F.relu(self.encode_spatial_1(encoded_boxes_pair))))
            # element-wise multiply with prod_rep
            prod_rep = prod_rep * spatial_rep

        if self.use_vision:
            vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
            if self.limit_vision:
                # exact value TBD
                prod_rep = torch.cat((prod_rep[:,:2048] * vr[:,:2048], prod_rep[:,2048:]), 1)
            else:
                prod_rep = prod_rep * vr

        if self.use_tanh:
            prod_rep = F.tanh(prod_rep)

        result.rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
                result.obj_preds[rel_inds[:, 1]],
                result.obj_preds[rel_inds[:, 2]],
            ), 1))

        if self.training and (not self.rl_train):
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)

        if not self.rl_train:
            return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep, gt_boxes, gt_classes, gt_rels)
        else:
            return result, filter_dets(bboxes, result.obj_scores, result.obj_preds, rel_inds[:, 1:], rel_rep, gt_boxes, gt_classes, gt_rels)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
