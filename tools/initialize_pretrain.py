# Copyright (c) OpenMMLab. All rights reserved.
"""Reshape the classification and regression layer for novel classes.

The bbox head from base training only supports `num_base_classes` prediction,
while in few shot fine-tuning it need to handle (`num_base_classes` +
`num_novel_classes`) classes. Thus, the layer related to number of classes
need to be reshaped.

The original implementation provides three ways to reshape the bbox head:

    - `combine`: combine two bbox heads from different models, for example,
        one model is trained with base classes data and another one is
        trained with novel classes data only.
    - `remove`: remove the final layer of the base model and the weights of
        the removed layer can't load from the base model checkpoint and
        will use random initialized weights for few shot fine-tuning.
    - `random_init`: create a random initialized layer (`num_base_classes` +
        `num_novel_classes`) and copy the weights of base classes from the
        base model.

Temporally, we only use this script in FSCE and TFA with `random_init`.
This part of code is modified from
https://github.com/ucbdrive/few-shot-object-detection/.

Example:
    # VOC base model
    python3 -m tools.detection.misc.initialize_bbox_head \
        --src1 work_dirs/tfa_r101_fpn_voc-split1_base-training/latest.pth \
        --method random_init \
        --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training
    # COCO base model
    python3 -m tools.detection.misc.initialize_bbox_head \
        --src1 work_dirs/tfa_r101_fpn_coco_base-training/latest.pth \
        --method random_init \
        --coco \
        --save-dir work_dirs/tfa_r101_fpn_coco_base-training
"""
import argparse
import os

import torch
from mmcv.runner.utils import set_random_seed


DOTA_SIZE = 17

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, help='Path to the main checkpoint')
    parser.add_argument(
        '--save-dir', type=str, default=None, help='Save directory')
    parser.add_argument(
        '--tar-name',
        type=str,
        default='base_model',
        help='Name of the new checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hbb', action='store_true', help='For HBB dataset')
    parser.add_argument('--save', action='store_true', help='For HBB dataset')
    return parser.parse_args()



def main():
    args = parse_args()
    set_random_seed(args.seed)
    ckpt = torch.load(args.src1)
    save_name = args.src1[:-4] + '_ready.pth'
    save_dir = '~/FSOD_remote/'
    save_path = save_name
    os.makedirs(save_dir, exist_ok=True)
    
    keys_rsp = [
    'mmdet_version', # 0 
    'CLASSES', #1
    'env_info', #2
    'config', #3
    'seed', #4
    'exp_name', #5
    'epoch', #6
    'iter',#7
    'mmcv_version',
    'time',#9
    'hook_msgs']
    print(ckpt['state_dict'].keys())
    ckpt2 = {}
    #if 'model' in ckpt.keys():
    #    ckpt['model'].pop('fc.bias', None)
    #    ckpt['model'].pop('fc.weight', None)
    #    ckpt2['meta'] = {
    #        'optimizer': ckpt['optimizer'],
    #        'lr_scheduler': ckpt['lr_scheduler'], 
    #        'max_accuracy': ckpt['max_accuracy'], 
    #        'epoch': ckpt['epoch'],
    #        'config': ckpt['config']
    #    }
    #    ckpt2['state_dict'] = ckpt['model']

    if args.save and False: 
        torch.save(ckpt2, save_path)
        print(f'save changed checkpoint to {save_path}')


if __name__ == '__main__':
    main()
