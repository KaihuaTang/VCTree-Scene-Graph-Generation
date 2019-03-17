import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
import numpy as np

from config import BOX_SCALE, IM_SCALE
from lib.fpn.box_utils import bbox_overlaps, bbox_intersections, center_size

class TreeLSTM_IO(object):
    def __init__(self, hidden_tensor, order_tensor, order_count, dists_tensor, commitments_tensor, dropout_mask):
        self.hidden = hidden_tensor # Float tensor [num_obj, self.out_dim]
        self.order = order_tensor # Long tensor [num_obj]
        self.order_count = order_count # int
        self.dists = dists_tensor # FLoat tensor [num_obj, len(self.classes)]
        self.commitments = commitments_tensor
        self.dropout_mask = dropout_mask

def get_overlap_info(im_inds, box_priors):
    """
    input:
        im_inds: [num_object]
        box_priors: [number_object, 4]
    output: [number_object, 6]
        number of overlapped obj (self not included)
        sum of all intersection area (self not included)
        sum of IoU (Intersection over Union)
        average of all intersection area (self not included)
        average of IoU (Intersection over Union)
        roi area
    """
    # generate forest
    num_obj = box_priors.shape[0]
    inds_offset = (im_inds * 1000).view(-1,1).expand(box_priors.shape)
    offset_box = box_priors + inds_offset.float()
    intersection = bbox_intersections(offset_box, offset_box) 
    overlap = bbox_overlaps(offset_box, offset_box)
    # [obj_num, obj_num], diagonal elements should been removed
    reverse_eye = Variable(1.0 - torch.eye(num_obj).float().cuda())
    intersection = intersection * reverse_eye
    overlap = overlap * reverse_eye
    box_area = bbox_area(offset_box)
    # generate input feat
    boxes_info = Variable(torch.FloatTensor(num_obj, 6).zero_().cuda()) # each obj has how many overlaped objects

    for obj_idx in range(num_obj):
            boxes_info[obj_idx, 0] = torch.nonzero(intersection[obj_idx]).numel()
            boxes_info[obj_idx, 1] = intersection[obj_idx].view(-1).sum() / float(IM_SCALE * IM_SCALE)
            boxes_info[obj_idx, 2] = overlap[obj_idx].view(-1).sum()
            boxes_info[obj_idx, 3] = boxes_info[obj_idx, 1] / (boxes_info[obj_idx, 0] + 1e-9)
            boxes_info[obj_idx, 4] = boxes_info[obj_idx, 2] /  (boxes_info[obj_idx, 0] + 1e-9)
            boxes_info[obj_idx, 5] = box_area[obj_idx] / float(IM_SCALE * IM_SCALE)
    
    return boxes_info, intersection

def get_box_info(boxes):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    return torch.cat((boxes / float(IM_SCALE), center_size(boxes) / float(IM_SCALE)), 1)


def get_box_pair_info(box1, box2):
    """
    input: 
        box1 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        box2 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    output: 
        32-digits: [box1, box2, unionbox, intersectionbox]
    """
    # union box
    unionbox = box1[:,:4].clone()
    unionbox[:, 0] = torch.min(box1[:, 0], box2[:, 0])
    unionbox[:, 1] = torch.min(box1[:, 1], box2[:, 1])
    unionbox[:, 2] = torch.max(box1[:, 2], box2[:, 2])
    unionbox[:, 3] = torch.max(box1[:, 3], box2[:, 3])
    union_info = get_box_info(unionbox)

    # intersection box
    intersextion_box = box1[:,:4].clone()
    intersextion_box[:, 0] = torch.max(box1[:, 0], box2[:, 0])
    intersextion_box[:, 1] = torch.max(box1[:, 1], box2[:, 1])
    intersextion_box[:, 2] = torch.min(box1[:, 2], box2[:, 2])
    intersextion_box[:, 3] = torch.min(box1[:, 3], box2[:, 3])
    case1 = torch.nonzero(intersextion_box[:, 2].contiguous().view(-1) < intersextion_box[:, 0].contiguous().view(-1)).view(-1)
    case2 = torch.nonzero(intersextion_box[:, 3].contiguous().view(-1) < intersextion_box[:, 1].contiguous().view(-1)).view(-1)
    intersextion_info = get_box_info(intersextion_box)
    if case1.numel() > 0:
        intersextion_info[case1, :] = 0
    if case2.numel() > 0:
        intersextion_info[case2, :] = 0

    return torch.cat((box1, box2, union_info, intersextion_info), 1)


def bbox_area(gt_boxes): 
    """
    gt_boxes: (K, 4) ndarray of float

    area: (k)
    """
    K = gt_boxes.size(0)
    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(K)

    return gt_boxes_area

def bbox_center(gt_boxes):
    """
    gt_boxes: (K, 4) ndarray of float

    center: (k, 2)
    """
    center = (gt_boxes[:, 2:] + gt_boxes[:, :2]) / 2.0
    assert(center.shape[1] == 2)
    return center

def print_tree(tree):
    if tree is None:
        return
    if(tree.left_child is not None):
        print_node(tree.left_child)
    if(tree.right_child is not None):
        print_node(tree.right_child)

    print_tree(tree.left_child)
    print_tree(tree.right_child)

    return
    

def print_node(tree):
    print(' depth: ', tree.depth(), end="")
    print(' label: ', tree.label, end="")
    print(' index: ', int(tree.index), end="")
    print(' score: ', tree.score(), end="")
    print(' center_x: ', tree.center_x)
