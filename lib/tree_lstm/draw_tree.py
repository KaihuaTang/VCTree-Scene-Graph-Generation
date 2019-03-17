import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from skimage.transform import resize
from config import IM_SCALE

tree_max_depth = 9

label_to_text = {"-1":"unknown-1", "0":"unknown-0", "1": "airplane", "2": "animal", "3": "arm", "4": "bag", "5": "banana", "6": "basket", "7": "beach", "8": "bear", "9": "bed", "10": "bench", "11": "bike", "12": "bird", "13": "board", "14": "boat", "15": "book", "16": "boot", "17": "bottle", "18": "bowl", "19": "box", "20": "boy", "21": "branch", "22": "building", "23": "bus", "24": "cabinet", "25": "cap", "26": "car", "27": "cat", "28": "chair", "29": "child", "30": "clock", "31": "coat", "32": "counter", "33": "cow", "34": "cup", "35": "curtain", "36": "desk", "37": "dog", "38": "door", "39": "drawer", "40": "ear", "41": "elephant", "42": "engine", "43": "eye", "44": "face", "45": "fence", "46": "finger", "47": "flag", "48": "flower", "49": "food", "50": "fork", "51": "fruit", "52": "giraffe", "53": "girl", "54": "glass", "55": "glove", "56": "guy", "57": "hair", "58": "hand", "59": "handle", "60": "hat", "61": "head", "62": "helmet", "63": "hill", "64": "horse", "65": "house", "66": "jacket", "67": "jean", "68": "kid", "69": "kite", "70": "lady", "71": "lamp", "72": "laptop", "73": "leaf", "74": "leg", "75": "letter", "76": "light", "77": "logo", "78": "man", "79": "men", "80": "motorcycle", "81": "mountain", "82": "mouth", "83": "neck", "84": "nose", "85": "number", "86": "orange", "87": "pant", "88": "paper", "89": "paw", "90": "people", "91": "person", "92": "phone", "93": "pillow", "94": "pizza", "95": "plane", "96": "plant", "97": "plate", "98": "player", "99": "pole", "100": "post", "101": "pot", "102": "racket", "103": "railing", "104": "rock", "105": "roof", "106": "room", "107": "screen", "108": "seat", "109": "sheep", "110": "shelf", "111": "shirt", "112": "shoe", "113": "short", "114": "sidewalk", "115": "sign", "116": "sink", "117": "skateboard", "118": "ski", "119": "skier", "120": "sneaker", "121": "snow", "122": "sock", "123": "stand", "124": "street", "125": "surfboard", "126": "table", "127": "tail", "128": "tie", "129": "tile", "130": "tire", "131": "toilet", "132": "towel", "133": "tower", "134": "track", "135": "train", "136": "tree", "137": "truck", "138": "trunk", "139": "umbrella", "140": "vase", "141": "vegetable", "142": "vehicle", "143": "wave", "144": "wheel", "145": "window", "146": "windshield", "147": "wing", "148": "wire", "149": "woman", "150": "zebra"}

def draw_tree_region_v2(tree, image, example_id, pred_labels):
    """
    tree: A tree structure
    image: origin image batch [batch_size, 3, IM_SIZE, IM_SIZE]
    output: a image with roi bbox, the color of box correspond to the depth of roi node
    """
    sample_image = image[tree.im_idx].view(image.shape[1:]).clone()
    sample_image = (revert_normalize(sample_image) * 255).int()
    sample_image = torch.clamp(sample_image, 0, 255)
    sample_image = sample_image.permute(1,2,0).contiguous().data.cpu().numpy().astype(dtype = np.uint8)
    sample_image = Image.fromarray(sample_image, 'RGB').convert("RGBA")

    draw = ImageDraw.Draw(sample_image)
    draw_box(draw, tree, pred_labels)
    
    sample_image.save('./output/example/'+str(example_id)+'_box'+'.png')

    #print('saved img ' + str(example_id))


def draw_box(draw, tree, pred_labels):
    x1,y1,x2,y2 = int(tree.box[0]), int(tree.box[1]), int(tree.box[2]), int(tree.box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline="red")
    draw.rectangle(((x1, y1), (x1+50, y1+10)), fill="red")
    node_label = int(pred_labels[int(tree.index)])
    draw.text((x1, y1), label_to_text[str(node_label)])

    if (tree.left_child is not None):
        draw_box(draw, tree.left_child, pred_labels)
    if (tree.right_child is not None):
        draw_box(draw, tree.right_child, pred_labels)
    


def draw_tree_region(tree, image, example_id):
    """
    tree: A tree structure
    image: origin image batch [batch_size, 3, IM_SIZE, IM_SIZE]
    output: a image display regions in a tree structure
    """
    sample_image = image[tree.im_idx].view(image.shape[1:]).clone()
    sample_image = (revert_normalize(sample_image) * 255).int()
    sample_image = torch.clamp(sample_image, 0, 255)
    sample_image = sample_image.permute(1,2,0).contiguous().data.cpu().numpy().astype(dtype = np.uint8)

    global tree_max_depth

    depth = min(tree.max_depth(), tree_max_depth)
    tree_img = create_tree_img(depth, 64)
    tree_img = write_cell(sample_image, tree_img, (0,0,tree_img.shape[1], tree_img.shape[0]), tree, 64)

    im = Image.fromarray(sample_image, 'RGB')
    tree_img = Image.fromarray(tree_img, 'RGB')
    im.save('./output/example/'+str(example_id)+'_origin'+'.jpg')
    tree_img.save('./output/example/'+str(example_id)+'_tree'+'.jpg')

    if example_id % 200 == 0:
        print('saved img ' + str(example_id))

def write_cell(orig_img, tree_img, draw_box, tree, cell_size):
    """
    orig_img: original image
    tree_img: draw roi tree
    draw_box: the whole bbox used to draw this sub-tree [x1,y1,x2,y2]
    tree: a sub-tree
    cell_size: size of each roi
    """
    x1,y1,x2,y2 = draw_box
    if tree is None:
        return tree_img
    # draw
    roi = orig_img[int(tree.box[1]):int(tree.box[3]), int(tree.box[0]):int(tree.box[2]), :]
    roi = Image.fromarray(roi, 'RGB')
    roi = roi.resize((cell_size, cell_size))
    roi = np.array(roi)
    draw_x1 = int(max((x1+x2)/2 - cell_size/2, 0))
    draw_x2 = int(min(draw_x1 + cell_size, x2))
    draw_y1 = y1
    draw_y2 = min(y1 + cell_size, y2)
    tree_img[draw_y1:draw_y2, draw_x1:draw_x2,:] = roi[:draw_y2-draw_y1,:draw_x2-draw_x1,:]
    # recursive draw
    global tree_max_depth
    if (tree.left_child is not None) and tree.left_child.depth() <= tree_max_depth:
        tree_img = write_cell(orig_img, tree_img, (x1,draw_y2,int((x1+x2)/2),y2), tree.left_child, cell_size)
    if (tree.right_child is not None) and tree.right_child.depth() <= tree_max_depth:
        tree_img = write_cell(orig_img, tree_img, (int((x1+x2)/2),draw_y2,x2,y2), tree.right_child, cell_size)
    
    return tree_img

def create_tree_img(depth, cell_size):
    height = cell_size * (depth)
    width = cell_size * (2**(depth-1))
    return np.zeros((height,width,3)).astype(dtype=np.uint8)

def revert_normalize(image):
    image[0,:,:] = image[0,:,:] * 0.229
    image[1,:,:] = image[1,:,:] * 0.224
    image[2,:,:] = image[2,:,:] * 0.225

    image[0,:,:] = image[0,:,:] + 0.485
    image[1,:,:] = image[1,:,:] + 0.456
    image[2,:,:] = image[2,:,:] + 0.406

    return image

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
