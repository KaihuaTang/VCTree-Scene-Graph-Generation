import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from lib.word_vectors import obj_edge_vectors
import numpy as np
from config import IM_SCALE, ROOT_PATH
import random

from lib.tree_lstm.def_tree import ArbitraryTree

def graph_to_trees(co_occour_prob, rel_labels, obj_labels):
    """
    Generate arbitrary trees according to the ground truth graph

    co_occour: [num_obj_classes, num_obj_classes]
    rel_labels: [image_index, i, j, rel_label], i is not j, i & j are under same image
    obj_labels: [image_index, obj_label]

    output: [forest]
    """
    num_nodes = obj_labels.shape[0]
    num_edges = rel_labels.shape[0]
    output = []

    # calculate the score of each edge 
    edge_scores = rel_labels.float().clone()
    for i in range(num_edges):
        if int(rel_labels[i, 3]) == 0:
            edge_scores[i, 3] = 0
        else:
            sub_id = int(obj_labels[int(rel_labels[i, 1]), 1]) - 1
            obj_id = int(obj_labels[int(rel_labels[i, 2]), 1]) - 1
            edge_scores[i, 3] = co_occour_prob[sub_id, obj_id]

    # generate score map (num_obj * num_obj)
    score_map = np.zeros((num_nodes, num_nodes))
    for i in range(num_edges):
        sub_id = int(edge_scores[i, 1])
        obj_id = int(edge_scores[i, 2])
        score_map[sub_id, obj_id] = float(score_map[sub_id, obj_id]) + float(edge_scores[i, 3])
        score_map[obj_id, sub_id] = float(score_map[obj_id, sub_id]) + float(edge_scores[i, 3])

    # calculate the score of each node
    node_scores = obj_labels.float().clone()
    node_scores[:,1] = 0
    for i in range(num_edges):
        node_id_1 = int(edge_scores[i, 1])
        node_id_2 = int(edge_scores[i, 2])
        node_scores[node_id_1, 1] = float(node_scores[node_id_1, 1]) + float(edge_scores[i, 3])
        node_scores[node_id_2, 1] = float(node_scores[node_id_2, 1]) + float(edge_scores[i, 3])

    # generate arbitrary tree
    group_id = 0
    im_inds = obj_labels[:, 0].contiguous()
    while(torch.nonzero(im_inds == group_id).numel() > 0):
        # generate node container
        node_container = []
        picked_list = torch.nonzero(im_inds == group_id).view(-1)
        for idx in picked_list:
            node_container.append(ArbitraryTree(idx, node_scores[int(idx), 1]))
        # use virtual root (entire image), index is -1, score is almost infinity
        tree_root = ArbitraryTree(-1, 10e10, is_root=True)
        # node insert order
        node_order = 0
        # find first & best node to start
        best_node = find_best_node(node_container)
        tree_root.add_child(best_node)
        best_node.node_order = node_order
        node_order += 1
        node_container.remove(best_node)
        # generate tree
        while(len(node_container) > 0):
            best_depend_node = None
            best_insert_node = None
            best_score = -1
            for i in range(len(node_container)):
                best_score, best_depend_node, best_insert_node = \
                    tree_root.search_best_insert(score_map, best_score, node_container[i], best_depend_node, best_insert_node)

            # if not in current tree, add to root, else insert
            if best_score == 0:
                best_node = find_best_node(node_container)
                tree_root.add_child(best_node)
                best_node.node_order = node_order
                node_order += 1
                node_container.remove(best_node)
            else:
                best_depend_node.add_child(best_insert_node)
                best_insert_node.node_order = node_order
                node_order += 1
                node_container.remove(best_insert_node)

        # add tree to forest
        output.append(tree_root)
        # next image
        group_id += 1
    
    return output
    

def arbitraryForest_to_biForest(forest):
    """
    forest: a set of arbitrary Tree
    output: a set of corresponding binary Tree
    """
    output = []
    for i in range(len(forest)):
        result_tree = arTree_to_biTree(forest[i])
        # make sure they are equivalent tree
        # assert(result_tree.get_total_child() == forest[i].get_total_child())
        output.append(result_tree)
        
    return output


def arTree_to_biTree(arTree):
    root_node = arTree.generate_bi_tree()
    arNode_to_biNode(arTree, root_node)

    return root_node

def arNode_to_biNode(arNode, biNode):
    if arNode.get_child_num() >= 1:
        new_bi_node = arNode.children[0].generate_bi_tree()
        biNode.add_left_child(new_bi_node)
        arNode_to_biNode(arNode.children[0], biNode.left_child)

    if arNode.get_child_num() > 1:
        current_bi_node = biNode.left_child
        for i in range(arNode.get_child_num() - 1):
            new_bi_node = arNode.children[i+1].generate_bi_tree()
            current_bi_node.add_right_child(new_bi_node)
            current_bi_node = current_bi_node.right_child
            arNode_to_biNode(arNode.children[i+1], current_bi_node)

def find_best_node(node_container):
    max_node_score = -1 
    best_node = None
    for i in range(len(node_container)):
        if node_container[i].score > max_node_score:
            max_node_score = node_container[i].score
            best_node = node_container[i]
    return best_node




