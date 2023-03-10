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
        '--save_dir', type=str, default=None, help='Save directory')
    parser.add_argument(
        '--tar-name',
        type=str,
        default='base_model',
        help='Name of the new checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hbb', action='store_true', help='For HBB dataset')
    parser.add_argument('--queue_length', type=int, default=0, help='For HBB dataset')
    parser.add_argument('--queue_save', action='store_true', help='For HBB dataset')
    parser.add_argument('--per_class', action='store_true', help='For HBB dataset')
    parser.add_argument('--nobg', action='store_true', help='For HBB dataset')
    parser.add_argument('--nobg_save', action='store_true', help='For HBB dataset')


    parser.add_argument('--base_withbg', action='store_true', help='For HBB dataset')
    parser.add_argument('--save', action='store_true', help='For HBB dataset')
    parser.add_argument('--queue', type=str, default=None, help='For HBB dataset')
    return parser.parse_args()



def main():
    args = parse_args()
    set_random_seed(args.seed)
    ckpt = torch.load(args.src1)
    save_name = args.src1[:-4] + '_ready.pth'
    save_dir = '~/FSOD_remote/'
    save_path = save_name
    #os.makedirs(save_dir, exist_ok=True)
    queue_res = ckpt['state_dict']['roi_head.bbox_head.queue_res']
    queue_trg = ckpt['state_dict']['roi_head.bbox_head.queue_trg']
    queue_iou = ckpt['state_dict']['roi_head.bbox_head.queue_iou']
    with open('per_class_features.p', 'rb') as fp:
            data = pickle.load(fp)
    num_base_classes = 15
    trg_queue_length = 126
    num_novel_classes = 20
    
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
        print(nbr_rep, trg_nbr_rep)

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
        
        for ii in ckpt2.keys():
            tpp = ckpt2[ii]['trg']
            #print(tpp)
        
        if args.nobg_save:
            data_name = 'Random_Queue_from_'+save_pickle_name+'_FT_nobg.p'
        else:
            data_name = 'Random_Queue_from_'+save_pickle_name+'_FT_withbg.p'
        #print(f' saved queue to:{data_name}')
        #with open(data_name, 'wb') as fp:
        #    pickle.dump(ckpt2, fp, protocol=pickle.HIGHEST_PROTOCOL)

    elif 'class' in args.queue:

        trg_nbr_rep = trg_queue_length//(num_novel_classes)
        print(nbr_rep, trg_nbr_rep)

        for class_id in range(num_base_classes): 
            tmp = []
            for k in range(nbr_rep):
                tmp.append(queue_res[k*(num_base_classes)+class_id])
            if tostack:
                tmp = torch.stack(tmp).mean(dim=0)  
            else:
                if trg_nbr_rep < nbr_rep:
                    #rnd_id = [i for i in range(len(tmp))]
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
        
        for ii in ckpt2.keys():
            tpp = ckpt2[ii]
            #print(ii)
        if args.nobg_save:
            data_name = 'Queue_perclass_from_'+save_pickle_name+'_FT_nobg.p'
        else:
            data_name = 'Queue_perclass_from_'+save_pickle_name+'_FT_withbg.p'
    data_name = 'queue_dict/'+data_name
    print(f' saved queue to:{data_name}')
    with open(data_name, 'wb') as fp:
        pickle.dump(ckpt2, fp, protocol=pickle.HIGHEST_PROTOCOL)



    if False:
        ckpt2 = {}
        save_pickle_name = args.src1[10:-39]

        trg_queue_length = 126
        num_base_classes = num_base_classes+1
        num_novel_classes = num_novel_classes+1

        nbr_rep = (queue_trg.shape[0])//(num_base_classes)
        trg_nbr_rep = trg_queue_length//(num_novel_classes)
        print(nbr_rep, trg_nbr_rep)
        for class_id in range(num_base_classes):
            tmp = []
            for k in range(nbr_rep):
                tmp.append(queue_res[k*(num_base_classes)+class_id])
            if tostack:
                tmp = torch.stack(tmp).mean(dim=0)  
            else:
                if trg_nbr_rep < nbr_rep:
                    #rnd_id = [i for i in range(len(tmp))]
                    rnd_pick = random.choices(tmp, k=trg_nbr_rep)
                    print(len(tmp), tmp[0].shape)
                    print(len(rnd_pick), rnd_pick[0].shape)
                    tmp = torch.stack(rnd_pick)
                else:
                    tmp = torch.stack(tmp)
            if class_id == num_base_classes-1:
                ckpt2[str(num_novel_classes-1)] = tmp
            else:
                ckpt2[str(class_id)] = tmp
        if trg_nbr_rep < nbr_rep:
            save_pickle_name += '_stack_rnd_select_' + str(trg_nbr_rep)
        elif tostack: 
                save_pickle_name += '_mean'
        elif trg_nbr_rep >= nbr_rep:
            save_pickle_name += '_stacked_' + str(trg_nbr_rep)
            
        print(ckpt2.keys(), ckpt2['0'].shape)
        #data_name = 'V2_per_class_features_from_'+save_pickle_name+'.p'
        #print(f' saved queue to:{data_name}')
        #with open(data_name, 'wb') as fp:
        #    pickle.dump(ckpt2, fp, protocol=pickle.HIGHEST_PROTOCOL)

    
    if False:

        reshaped_trg = queue_trg.clone().reshape(-1)
        reshaped_iou = queue_iou.clone().reshape(-1)
        reshaped_res = queue_res.clone().reshape(-1, queue_res.shape[2])
        queue_dict = {}
        q_len = args.queue_length*256
        if args.per_class:
            for class_id in range(num_base_classes+1):
                
                if class_id in reshaped_trg.unique():
                    msk = reshaped_trg == class_id
                    iou_msk = reshaped_iou[msk] == reshaped_iou[msk].max()
                    res = reshaped_res[msk][iou_msk].float().mean(dim=0)
                    iou = reshaped_iou[msk][iou_msk].float().mean()
                    trg = reshaped_trg[msk][iou_msk].float().mean().int()
                    print(f' shapes of class {class_id}: res {res.shape}, trg {trg}, iou {iou}')

                    if class_id == num_base_classes:
                        tmp_dict ={'results':res, 'trg':trg, 'iou':iou} 
                        queue_dict['bg']=tmp_dict
                    else:
                        tmp_dict ={'results':res, 'trg':trg, 'iou':iou} 
                        queue_dict[class_id]=tmp_dict
        else:
            if q_len <= reshaped_iou.shape[0]:
                iou_msk = torch.topk(reshaped_iou, q_len).indices
                res = reshaped_res[iou_msk]
                trg = reshaped_trg[iou_msk]
                iou = reshaped_iou[iou_msk]
                print(f' shapes of class {q_len}: res {res.shape}, trg {trg.shape}, iou {iou.shape}')
            else:
                res = reshaped_res
                trg = reshaped_trg
                iou = reshaped_iou
                print(f'res {res.shape}, trg {trg.shape}, iou {iou.shape}')
            for class_id in trg.unique():
                msk = trg == class_id
                print(f'res {res[msk].shape}, trg {trg[msk].shape}, iou {iou[msk].shape}')
                tmp_trg = trg[msk].reshape(-1,256)
                tmp_iou = iou[msk].reshape(-1,256)
                tmp_res = res[msk].reshape(tmp_iou.shape[0],256,-1)
                print(f'res {tmp_res.shape}, trg {tmp_trg.shape}, iou {tmp_iou.shape}')
                if class_id == num_base_classes:
                    tmp_dict ={'results':tmp_res, 'trg':tmp_trg, 'iou':tmp_iou} 
                    queue_dict['bg']=tmp_dict
                else:
                    tmp_dict ={'results':tmp_res, 'trg':tmp_trg, 'iou':tmp_iou} 
                    queue_dict[class_id]=tmp_dict
                

        if args.queue_save:
            with open(args.save_dir, 'wb') as fp:
                pickle.dump(queue_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        #ckpt2 = {}
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

def main_nobg():
    args = parse_args()
    set_random_seed(args.seed)
    ckpt = torch.load(args.src1)
    save_name = args.src1[:-4] + '_ready.pth'
    save_dir = '~/FSOD_remote/'
    save_path = save_name
    #os.makedirs(save_dir, exist_ok=True)
    queue_res = ckpt['state_dict']['roi_head.bbox_head.queue_res']
    queue_trg = ckpt['state_dict']['roi_head.bbox_head.queue_trg']
    queue_iou = ckpt['state_dict']['roi_head.bbox_head.queue_iou']
    with open('per_class_features.p', 'rb') as fp:
            data = pickle.load(fp)
    num_base_classes = 15
    trg_queue_length = 120
    num_novel_classes = 20
    tostack = False

    ckpt2 = {}
    save_pickle_name = args.src1[10:-39]
    
    print(queue_trg, queue_trg.shape, queue_iou.shape, queue_res.shape)
    queue_res = queue_res.reshape(trg_queue_length, -1, queue_res.shape[2])
    queue_trg = queue_trg.reshape(trg_queue_length, -1)
    queue_iou = queue_iou.reshape(trg_queue_length, -1)

    print(dict().shape)
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
                #rnd_id = [i for i in range(len(tmp))]
                rnd_pick = random.choices(tmp, k=trg_nbr_rep)
                print(len(tmp), tmp[0].shape)
                print(len(rnd_pick), rnd_pick[0].shape)
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

    
    if False:

        reshaped_trg = queue_trg.clone().reshape(-1)
        reshaped_iou = queue_iou.clone().reshape(-1)
        reshaped_res = queue_res.clone().reshape(-1, queue_res.shape[2])
        queue_dict = {}
        q_len = args.queue_length*256
        if args.per_class:
            for class_id in range(num_base_classes+1):
                
                if class_id in reshaped_trg.unique():
                    msk = reshaped_trg == class_id
                    iou_msk = reshaped_iou[msk] == reshaped_iou[msk].max()
                    res = reshaped_res[msk][iou_msk].float().mean(dim=0)
                    iou = reshaped_iou[msk][iou_msk].float().mean()
                    trg = reshaped_trg[msk][iou_msk].float().mean().int()
                    print(f' shapes of class {class_id}: res {res.shape}, trg {trg}, iou {iou}')

                    if class_id == num_base_classes:
                        tmp_dict ={'results':res, 'trg':trg, 'iou':iou} 
                        queue_dict['bg']=tmp_dict
                    else:
                        tmp_dict ={'results':res, 'trg':trg, 'iou':iou} 
                        queue_dict[class_id]=tmp_dict
        else:
            if q_len <= reshaped_iou.shape[0]:
                iou_msk = torch.topk(reshaped_iou, q_len).indices
                res = reshaped_res[iou_msk]
                trg = reshaped_trg[iou_msk]
                iou = reshaped_iou[iou_msk]
                print(f' shapes of class {q_len}: res {res.shape}, trg {trg.shape}, iou {iou.shape}')
            else:
                res = reshaped_res
                trg = reshaped_trg
                iou = reshaped_iou
                print(f'res {res.shape}, trg {trg.shape}, iou {iou.shape}')
            for class_id in trg.unique():
                msk = trg == class_id
                print(f'res {res[msk].shape}, trg {trg[msk].shape}, iou {iou[msk].shape}')
                tmp_trg = trg[msk].reshape(-1,256)
                tmp_iou = iou[msk].reshape(-1,256)
                tmp_res = res[msk].reshape(tmp_iou.shape[0],256,-1)
                print(f'res {tmp_res.shape}, trg {tmp_trg.shape}, iou {tmp_iou.shape}')
                if class_id == num_base_classes:
                    tmp_dict ={'results':tmp_res, 'trg':tmp_trg, 'iou':tmp_iou} 
                    queue_dict['bg']=tmp_dict
                else:
                    tmp_dict ={'results':tmp_res, 'trg':tmp_trg, 'iou':tmp_iou} 
                    queue_dict[class_id]=tmp_dict
                

        if args.queue_save:
            with open(args.save_dir, 'wb') as fp:
                pickle.dump(queue_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        if args.save and False: 
            torch.save(ckpt2, save_path)
            print(f'save changed checkpoint to {save_path}')


if __name__ == '__main__':
    args = parse_args()
    if args.nobg:
        main_nobg()
    else:
        main()
