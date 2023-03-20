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
import pickle 

import random
import torch
from mmcv.runner.utils import set_random_seed


DOTA_SIZE = 17

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, default='work_dirs/TrueFT_cls_loss_10shots_contrastive_separate_nobbox_basePL_nolight_rnd_w10_nomean_more2_nobg/latest.pth', help='Path to the main checkpoint')
    parser.add_argument(
        '--save_dir', type=str, default='~/FSOD_remote/', help='Save directory')
    parser.add_argument(
        '--tar-name',
        type=str,
        default='base_model',
        help='Name of the new checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hbb', action='store_true', help='For HBB dataset')
    parser.add_argument('--nobg', action='store_true', help='Boolean to define if the fine-tuning queue is including background')
    parser.add_argument('--nobg_save', action='store_true', help='Boolean to define if the fine-tuning queue is including background')

    parser.add_argument('--target_queue_length', type=int, default=126, help='Targe queue length')
    parser.add_argument('--nbr_base_class', type=int, default=15, help='Number of classes for base training')
    parser.add_argument('--nbr_ft_class', type=int, default=20, help='Number of classes for fine-tuning')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')


    parser.add_argument('--base_withbg', action='store_true', help='Boolean to define if the base queue is including background')
    parser.add_argument('--queue', choices=['random', 'class'], default='random', help='Queue design for fine-tuning. Either random or class.')
    
    return parser.parse_args()



def main():
    """
    Create a queue to load for the fine-tuning based on the base training queue
    """
    args = parse_args()
    set_random_seed(args.seed)
    ckpt = torch.load(args.src1)
    save_name = args.src1[:-4] + '_ready.pth'
    save_dir = args.save_dir
    save_path = save_name
    os.makedirs(save_dir, exist_ok=True)
    queue_res = ckpt['state_dict']['roi_head.bbox_head.queue_res']
    queue_trg = ckpt['state_dict']['roi_head.bbox_head.queue_trg']
    queue_iou = ckpt['state_dict']['roi_head.bbox_head.queue_iou']

    num_base_classes = args.nbr_base_class
    trg_queue_length = args.target_queue_length
    num_novel_classes = args.nbr_ft_class
    
    tostack = False
    ckpt2 = {}
    save_pickle_name = args.src1[10:-39]
    if args.base_withbg:
        num_base_classes = num_base_classes+1

    num_novel_classes = num_novel_classes+1
    nbr_rep = (queue_trg.shape[0])//(num_base_classes)
    cont_shape = queue_res.shape[1]
    if 'random' in args.queue:

        trg_nbr_rep = trg_queue_length
        for id_base in range(queue_trg.shape[0]): 
            tmp_dict = {}
            tmp_dict['results'] = queue_res[id_base]
            tmp_dict['iou'] = queue_iou[id_base]
            if queue_trg[id_base] == num_base_classes-1:
                if args.nobg_save:
                    continue
                tmp_dict['trg'] = torch.tensor(float(num_novel_classes))
            else:
                tmp_dict['trg'] = queue_trg[id_base]
            ckpt2[id_base] = tmp_dict
        
        if args.nobg_save:
            data_name = 'Random_Queue_from_'+save_pickle_name+'_FT_nobg.p'
        else:
            data_name = 'Random_Queue_from_'+save_pickle_name+'_FT_withbg.p'

    elif 'class' in args.queue:

        trg_nbr_rep = trg_queue_length//(num_novel_classes)
        for class_id in range(num_base_classes): 
            tmp = []
            for k in range(nbr_rep):
                tmp.append(queue_res[k*(num_base_classes)+class_id])
            if tostack:
                tmp = torch.stack(tmp).mean(dim=0)  
            else:
                if trg_nbr_rep < nbr_rep:
                    rnd_pick = random.choices(tmp, k=trg_nbr_rep)
                    
                    tmp = torch.stack(rnd_pick)
                else:
                    tmp = torch.stack(tmp)
            if class_id == num_base_classes-1:
                if args.nobg_save:
                    continue
                ckpt2[str(num_novel_classes-1)] = tmp
            else:
                ckpt2[str(class_id)] = tmp
        
        if args.nobg_save:
            data_name = 'Queue_perclass_from_'+save_pickle_name+'_FT_nobg.p'
        else:
            data_name = 'Queue_perclass_from_'+save_pickle_name+'_FT_withbg.p'
    data_name = 'queue_dict/'+data_name
    print(f' saved queue to:{data_name}')
    with open(data_name, 'wb') as fp:
        pickle.dump(ckpt2, fp, protocol=pickle.HIGHEST_PROTOCOL)




def main_nobg():
    """
    Create a queue to load for the fine-tuning excluding bakcground base on the base training queue
    """
    args = parse_args()
    set_random_seed(args.seed)
    ckpt = torch.load(args.src1)
    save_name = args.src1[:-4] + '_ready.pth'
    save_dir = args.save_dir
    save_path = save_name
    os.makedirs(save_dir, exist_ok=True)
    queue_res = ckpt['state_dict']['roi_head.bbox_head.queue_res']
    queue_trg = ckpt['state_dict']['roi_head.bbox_head.queue_trg']
    queue_iou = ckpt['state_dict']['roi_head.bbox_head.queue_iou']

    num_base_classes = 15
    trg_queue_length = 120
    num_novel_classes = 20
    tostack = False

    ckpt2 = {}
    save_pickle_name = args.src1[10:-39]
    
    queue_res = queue_res.reshape(trg_queue_length, -1, queue_res.shape[2])
    queue_trg = queue_trg.reshape(trg_queue_length, -1)
    queue_iou = queue_iou.reshape(trg_queue_length, -1)

    nbr_rep = (queue_trg.shape[0])//(num_base_classes)
    trg_nbr_rep = trg_queue_length//(num_novel_classes)
    for class_id in range(num_base_classes):
        tmp = []
        for k in range(nbr_rep):
            tmp.append(queue_res[k*(num_base_classes)+class_id])
        if tostack:
            tmp = torch.stack(tmp).mean(dim=0)  
        else:
            if trg_nbr_rep < nbr_rep:
                rnd_pick = random.choices(tmp, k=trg_nbr_rep)
                tmp = torch.stack(rnd_pick)
            else:
                tmp = torch.stack(tmp)
        
        ckpt2[str(class_id)] = tmp

    if trg_nbr_rep < nbr_rep:
        save_pickle_name += '_stack_rnd_select_' + str(trg_nbr_rep)
    elif tostack: 
            save_pickle_name += '_mean'
    elif trg_nbr_rep >= nbr_rep:
        save_pickle_name += '_stacked_' + str(trg_nbr_rep)
    
    save_pickle_name += '_nobg'

    data_name = 'V2_per_class_features_from_'+save_pickle_name+'.p'
    print(f' saved queue to:{data_name}')
    with open(data_name, 'wb') as fp:
        pickle.dump(ckpt2, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parse_args()
    if args.nobg:
        main_nobg()
    else:
        main()
