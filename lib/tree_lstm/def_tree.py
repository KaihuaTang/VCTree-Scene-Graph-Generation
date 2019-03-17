import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
import numpy as np

from config import BOX_SCALE, IM_SCALE


class BasicBiTree(object):
    def __init__(self, idx, is_root=False):
        self.index = int(idx)
        self.is_root = is_root
        self.left_child = None
        self.right_child = None
        self.parent = None
        self.num_child = 0

    def add_left_child(self, child):
        if self.left_child is not None:
            print('Left child already exist')
            return
        child.parent = self
        self.num_child += 1
        self.left_child = child
    
    def add_right_child(self, child):
        if self.right_child is not None:
            print('Right child already exist')
            return
        child.parent = self
        self.num_child += 1
        self.right_child = child

    def get_total_child(self):
        sum = 0
        sum += self.num_child
        if self.left_child is not None:
            sum += self.left_child.get_total_child()
        if self.right_child is not None:
            sum += self.right_child.get_total_child()
        return sum

    def depth(self):
        if hasattr(self, '_depth'):
            return self._depth
        if self.parent is None:
            count = 1
        else:
            count = self.parent.depth() + 1
        self._depth = count
        return self._depth

    def max_depth(self):
        if hasattr(self, '_max_depth'):
            return self._max_depth
        count = 0
        if self.left_child is not None:
            left_depth = self.left_child.max_depth()
            if left_depth > count:
                count = left_depth
        if self.right_child is not None:
            right_depth = self.right_child.max_depth()
            if right_depth > count:
                count = right_depth
        count += 1
        self._max_depth = count
        return self._max_depth

    # by index
    def is_descendant(self, idx):
        left_flag = False
        right_flag = False
        # node is left child
        if self.left_child is not None:
            if self.left_child.index is idx:
                return True
            else:
                left_flag = self.left_child.is_descendant(idx)
        # node is right child
        if self.right_child is not None:
            if self.right_child.index is idx:
                return True
            else:
                right_flag = self.right_child.is_descendant(idx)
        # node is descendant
        if left_flag or right_flag:
            return True
        else:
            return False
            
    # whether input node is under left sub tree
    def is_left_descendant(self, idx):
        if self.left_child is not None:
            if self.left_child.index is idx:
                return True
            else:
                return self.left_child.is_descendant(idx)
        else:
            return False
    
    # whether input node is under right sub tree
    def is_right_descendant(self, idx):
        if self.right_child is not None:
            if self.right_child.index is idx:
                return True
            else:
                return self.right_child.is_descendant(idx)
        else:
            return False

    
class ArbitraryTree(object):
    def __init__(self, idx, score, label=-1, box=None, im_idx=-1, is_root=False):
        self.index = int(idx)
        self.is_root = is_root
        self.score = float(score)
        self.children = []
        self.label = label
        self.embeded_label = None
        self.box = box.view(-1) if box is not None else None #[x1,y1,x2,y2]
        self.im_idx = int(im_idx) # which image it comes from
        self.parent = None
        self.node_order = -1 # the n_th node added to the tree
    
    def generate_bi_tree(self):
        # generate a BiTree node, parent/child relationship are not inherited
        return BiTree(self.index, self.score, self.label, self.box, self.im_idx, self.is_root)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def print(self):
        print('index: ', self.index)
        print('node_order: ', self.node_order)
        print('num of child: ', len(self.children))
        for node in self.children:
            node.print()

    def find_node_by_order(self, order, result_node):
        if self.node_order == order:
            result_node = self
        elif len(self.children) > 0:
            for i in range(len(self.children)):
                result_node = self.children[i].find_node_by_order(order, result_node)
        
        return result_node
    
    def find_node_by_index(self, index, result_node):
        if self.index == index:
            result_node = self
        elif len(self.children) > 0:
            for i in range(len(self.children)):
                result_node = self.children[i].find_node_by_index(index, result_node)
                
        return result_node

    def search_best_insert(self, score_map, best_score, insert_node, best_depend_node, best_insert_node, ignore_root = True):
        if self.is_root and ignore_root:
            pass
        elif float(score_map[self.index, insert_node.index]) > float(best_score):
            best_score = score_map[self.index, insert_node.index]
            best_depend_node = self
            best_insert_node = insert_node
        
        # iteratively search child
        for i in range(self.get_child_num()):
            best_score, best_depend_node, best_insert_node = \
                self.children[i].search_best_insert(score_map, best_score, insert_node, best_depend_node, best_insert_node)

        return best_score, best_depend_node, best_insert_node

    def get_child_num(self):
        return len(self.children)
    
    def get_total_child(self):
        sum = 0
        num_current_child = self.get_child_num()
        sum += num_current_child
        for i in range(num_current_child):
            sum += self.children[i].get_total_child()
        return sum

# only support binary tree
class BiTree(BasicBiTree):
    def __init__(self, idx, node_score, label, box, im_idx, is_root=False):
        super(BiTree, self).__init__(idx, is_root)
        self.state_c = None
        self.state_h = None
        self.state_c_backward = None
        self.state_h_backward = None
        # used to select node
        self.node_score = float(node_score)
        self.label = label
        self.embeded_label = None
        self.box = box.view(-1) #[x1,y1,x2,y2]
        self.im_idx = int(im_idx) # which image it comes from

