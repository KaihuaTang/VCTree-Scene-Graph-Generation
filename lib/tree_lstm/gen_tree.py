import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from lib.word_vectors import obj_edge_vectors
import numpy as np
from config import IM_SCALE
import random

from lib.tree_lstm import tree_utils
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import block_orthogonal
from lib.tree_lstm.def_tree import ArbitraryTree
from config import LOG_SOFTMAX

class RLFeatPreprocessNet(nn.Module):
    def __init__(self, feature_size, embed_size, box_info_size, overlap_info_size, output_size):
        super(RLFeatPreprocessNet, self).__init__()
        self.feature_size = feature_size
        self.embed_size = embed_size
        self.box_info_size = box_info_size
        self.overlap_info_size = overlap_info_size
        self.output_size = output_size

        # linear layers
        self.resize_feat = nn.Linear(self.feature_size, int(output_size / 4))
        self.resize_embed = nn.Linear(self.embed_size, int(output_size / 4))
        self.resize_box = nn.Linear(self.box_info_size, int(output_size / 4))
        self.resize_overlap = nn.Linear(self.overlap_info_size, int(output_size / 4))

        # init
        self.resize_feat.weight.data.normal_(0, 0.001)
        self.resize_embed.weight.data.normal_(0, 0.01)
        self.resize_box.weight.data.normal_(0, 1)
        self.resize_overlap.weight.data.normal_(0, 1)
        self.resize_feat.bias.data.zero_()
        self.resize_embed.bias.data.zero_()
        self.resize_box.bias.data.zero_()
        self.resize_overlap.bias.data.zero_()

    def forward(self, obj_feat, obj_embed, box_info, overlap_info):
        resized_obj = self.resize_feat(obj_feat)
        resized_embed = self.resize_embed(obj_embed)
        resized_box = self.resize_box(box_info)
        resized_overlap = self.resize_overlap(overlap_info)

        output_feat = torch.cat((resized_obj, resized_embed, resized_box, resized_overlap), 1)
        return output_feat

def generate_forest(im_inds, gt_forest, pair_scores, box_priors, obj_label, use_rl_tree, is_training, mode):
    """
    generate a list of trees that covers all the objects in a batch
    im_inds: [obj_num]
    box_priors: [obj_num, (x1, y1, x2, y2)]
    pair_scores: [obj_num, obj_num]
    
    output: list of trees, each present a chunk of overlaping objects
    """
    output_forest = []  # the list of trees, each one is a chunk of overlapping objects
    num_obj = box_priors.shape[0]
    
    # node score: accumulate parent scores
    node_scores = pair_scores.mean(1).view(-1)
    # make forest
    group_id = 0

    gen_tree_loss_per_batch = []
    entropy_loss = []

    while(torch.nonzero(im_inds == group_id).numel() > 0):
        # select the nodes from the same image
        rl_node_container = []
        remain_index = []
        picked_list = torch.nonzero(im_inds == group_id).view(-1)
        root_idx = picked_list[-1]

        rl_root = ArbitraryTree(root_idx, node_scores[int(root_idx)], -1, box_priors[int(root_idx)], im_inds[int(root_idx)], is_root=True)

        # put all nodes into node container
        for idx in picked_list[:-1]:
            if obj_label is not None:
                label = int(obj_label[idx])
            else:
                label = -1
            new_node = ArbitraryTree(idx, node_scores[idx], label, box_priors[idx], im_inds[idx])
            rl_node_container.append(new_node)
            remain_index.append(int(idx))
        
        # iteratively generate tree
        rl_gen_tree(rl_node_container, pair_scores, node_scores, gen_tree_loss_per_batch, entropy_loss, rl_root, remain_index, (is_training and use_rl_tree), mode)

        output_forest.append(rl_root)
        group_id += 1

    return output_forest, gen_tree_loss_per_batch, entropy_loss
    

def rl_gen_tree(node_container, pair_scores, node_scores, gen_tree_loss_per_batch, entropy_loss, rl_root, remain_index, rl_training, mode):
    """
    use reinforcement learning to generate loss (without baseline tree)
    Calculate the log(pr) for each decision (not cross entropy)
    Step 1: Devide all nodes into left child container and right child container
    Step 2: From left child container and right child container, select their respective sub roots

    pair_scores: [obj_num, obj_num]
    node_scores: [obj_num]
    """
    num_nodes = len(node_container)
    # Step 0
    if  num_nodes == 0:
        return
    # Step 1
    select_node = []
    select_index = []
    select_node.append(rl_root)
    select_index.append(rl_root.index)

    if mode == 'predcls':
        first_score = node_scores[remain_index].contiguous().view(-1)
        _, inds = F.softmax(first_score, 0).max(0)
        first_node = node_container[int(inds)]
        rl_root.add_child(first_node)
        select_node.append(first_node)
        select_index.append(first_node.index)
        node_container.remove(first_node)
        remain_index.remove(first_node.index)

    not_sampled = True

    while len(node_container) > 0:
        wid = len(remain_index)
        select_index_var = Variable(torch.LongTensor(select_index).cuda())
        remain_index_var = Variable(torch.LongTensor(remain_index).cuda())
        select_score_map = torch.index_select( torch.index_select(pair_scores, 0, select_index_var), 1, remain_index_var ).contiguous().view(-1)
        #select_score_map = pair_scores[select_index][:,remain_index].contiguous().view(-1)
        if rl_training and not_sampled:
            dist = F.softmax(select_score_map, 0)
            greedy_id = dist.max(0)[1]
            best_id = torch.multinomial(dist, 1)[0]
            if int(greedy_id) != int(best_id):
                not_sampled = False
                if LOG_SOFTMAX:
                    prob = dist[best_id] + 1e-20
                else:
                    prob = select_score_map[best_id] + 1e-20
                gen_tree_loss_per_batch.append(prob.log())
            #neg_entropy = dist * (dist + 1e-20).log()
            #entropy_loss.append(neg_entropy.sum())
        else:
            best_score, best_id = select_score_map.max(0)
        depend_id = int(best_id) // wid
        insert_id = int(best_id) % wid
        best_depend_node = select_node[depend_id]
        best_insert_node = node_container[insert_id]
        best_depend_node.add_child(best_insert_node)

        select_node.append(best_insert_node)
        select_index.append(best_insert_node.index)
        node_container.remove(best_insert_node)
        remain_index.remove(best_insert_node.index)

        
        

    









