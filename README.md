# VCTree-Scene-Graph-Generation
Code for the Scene Graph Generation part of CVPR 2019 oral paper: "[Learning to Compose Dynamic Tree Structures for Visual Contexts][0]" 

UGLY CODE WARNING! UGLY CODE WARNING! UGLY CODE WARNING!

The code is directly modified from the project [rowanz/neural-motifs][1]. Most of the Codes about the proposed VCTree are located at lib/tree_lstm/*, and if you get any problem that cause you unable to run the project, you can check the issues under [rowanz/neural-motifs][1] first. 

# Dependencies
- You may follow these commands to establish the environments under Ubuntu system
```
Install Anaconda
conda update -n base conda
conda create -n motif pip python=3.6
conda install pytorch=0.3 torchvision cuda90 -c pytorch
bash install_package.sh
```

# Prepare Dataset and Setup

0. Please follow the [Instruction][2] under ./data/stanford_filtered/ to download the dateset and put them under proper locations. 

1. Update the config file with the dataset paths. Specifically:
    - Visual Genome (the VG_100K folder, image_data.json, VG-SGG.h5, and VG-SGG-dicts.json). See data/stanford_filtered/README.md for the steps I used to download these.
    - You'll also need to fix your PYTHONPATH: ```export PYTHONPATH=/home/rowan/code/scene-graph``` 

2. Compile everything. run ```make``` in the main directory: this compiles the Bilinear Interpolation operation for the RoIs.

3. Pretrain VG detection. The old version involved pretraining COCO as well, but we got rid of that for simplicity. Run ./scripts/pretrain_detector.sh
Note: You might have to modify the learning rate and batch size, particularly if you don't have 3 Titan X GPUs (which is what I used). [You can also download the pretrained detector checkpoint here.](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX) Note that, this detector model is the default initialization of all VCTree models, so when you download this checkpoint, you need to change the "-ckpt THE_PATH_OF_INITIAL_CHECKPOINT_MODEL" under ./scripts/train_vctreenet


# How to Train / Evaluation
0. Note that, most of the parameters are under config.py. The training stages and settings are manipulated through ./scripts/train_vctreenet.sh Each line of command in train_vctreenet.sh needs to manually indicate "-ckpt" model (initial parameters) and "-save_dir" the path to save model. Since we have hybrid learning strategy, each task predcls/sgcls/sgdet will have two options for supervised stage and reinformence finetuning stage, respectively, when iteratively switch the stage, the -ckpt should start with previous -save_dir. The first supervised stage will init with [detector checkpoint](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX) as mentioned above.

1. Train VG predicate classification (predcls) 
    - stage 1 (supervised stage of hybrid learning): run ./scripts/train_vctreenet.sh 5 
    - stage 2 (reinformence finetuning stage of hybrid learning): run ./scripts/train_vctreenet.sh 4 
    - (By default, it will run on GPU 2, you can modify CUDA_VISIBLE_DEVICES under train_vctreenet.sh). 
    - The model will be saved by the name "-save_dir checkpoints/THE_NAME_YOU_WILL_SAVE_THE_MODEL"

2. Train VG scene graph classification (sgcls)
    - stage 1 (supervised stage of hybrid learning): run ./scripts/train_vctreenet.sh 3 
    - stage 2 (reinformence finetuning stage of hybrid learning): run ./scripts/train_vctreenet.sh 2 
    - (By default, it will run on GPU 2, you can modify CUDA_VISIBLE_DEVICES under train_vctreenet.sh). 
    - The model will be saved by the name "-save_dir checkpoints/THE_NAME_YOU_WILL_SAVE_THE_MODEL"

3. Train VG scene graph detection (sgdet)
    - stage 1 (supervised stage of hybrid learning): run ./scripts/train_vctreenet.sh 1 
    - stage 2 (reinformence finetuning stage of hybrid learning): run ./scripts/train_vctreenet.sh 0 
    - (By default, it will run on GPU 2, you can modify CUDA_VISIBLE_DEVICES under train_vctreenet.sh). 
    - The model will be saved by the name "-save_dir checkpoints/THE_NAME_YOU_WILL_SAVE_THE_MODEL"

4. Evaluate predicate classification (predcls): 
    - run ./scripts/eval_models.sh 0
    - OR, You can simply download our predcls checkpoint: [VCTree/PredCls][3].

5. Evaluate scene graph classification (sgcls): 
    - run ./scripts/eval_models.sh 1
    - OR, You can simply download our sgcls checkpoint: [VCTree/SGCls][4].

6. Evaluate scene graph detection (sgdet): 
    - run ./scripts/eval_models.sh 2
    - OR, You can simply download our sgdet checkpoint: [VCTree/SGDET][5].


# If this paper/project inspires your work, pls cite our work:
```
@inproceedings{tang2018learning,
  title={Learning to Compose Dynamic Tree Structures for Visual Contexts},
  author={Tang, Kaihua and Zhang, Hanwang and Wu, Baoyuan and Luo, Wenhan and Liu, Wei},
  booktitle= "Conference on Computer Vision and Pattern Recognition",
  year={2019}
}
```

[0]: https://arxiv.org/abs/1812.01880
[1]: https://github.com/rowanz/neural-motifs
[2]: https://github.com/rowanz/neural-motifs/tree/master/data/stanford_filtered
[3]: PredCls_path
[4]: sgcls_path
[5]: sgdet_path