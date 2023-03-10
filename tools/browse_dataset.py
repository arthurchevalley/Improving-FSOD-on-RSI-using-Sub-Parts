# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import Sequence
from pathlib import Path
import torch
import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization.image import imshow_det_bboxes, draw_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root
from skimage import segmentation
import matplotlib.image
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default='/home/archeval/mmdetection/CATNet/mmdetection/plots/pres3/',
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument('--nBBOX_out', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    
    def skip_pipeline_steps(config):
        if 'pipeline' in config.keys():
            config['pipeline'] = [
                x for x in config.pipeline if x['type'] not in skip_type
            ]

    cfg = Config.fromfile(config_path)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def selected():
    from PIL import Image
    np.random.seed(44)
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))
    stop = 0
    contrastive_bool = False
    
    img_list = ['01420.jpg','15104.jpg','13001.jpg', '13236.jpg', '00318.jpg','12743.jpg', '02022.jpg', '14473.jpg', '14820.jpg', '05419.jpg','10541.jpg', '01233.jpg','12951.jpg','12856.jpg','06197.jpg','13324.jpg','']#'00001.jpg','00002.jpg','00003.jpg','00004.jpg','00005.jpg','00006.jpg', '00007.jpg']#'00003.jpg', '00006.jpg', '00156.jpg', '00478.jpg', '00746.jpg']
    #img_list = ['05357.jpg','04640.jpg','01816.jpg','02155.jpg','01407.jpg','00485.jpg','00932.jpg']
    img_list = []
    img_list2 = []
    for file_id, item_all in tqdm(enumerate(dataset)):
        if contrastive_bool:
            c_img = item_all['img_metas'].data
            item_three = [item_all['base'], item_all['novel'], item_all['augmented']]
            c_img = c_img['ori_filename']
        else:
            c_img = item_all
            name_img_meta = item_all['img_metas'].data
            c_img = name_img_meta['ori_filename']
            
            lbl = item_all['base']['gt_labels'].data
            if len(lbl.shape) > 1:
                lbl = lbl[0]
            else:
                lbl = lbl.item()
            if 'base' in item_all.keys():
                item_three = [item_all['base']]
            else:
                item_three = [item_all]
        ssave_id = 0
        if lbl not in img_list:
            ssave_id = 1
            img_list.append(lbl)
        elif lbl not in img_list2:
            ssave_id = 2
            img_list2.append(lbl)
        else:
            continue 
        #if c_img not in img_list:
        #    continue 
        #el
        if stop == 40:#len(img_list):
            break
        else:
            stop += 1
            
            name = {0:'base', 1:'novel', 2:'aug'}
            for id, item in enumerate(item_three):
                #if Path(item['filename']).name == img_list[0]:
                #    stop += 1
                #else:
                
                    #progress_bar.update()
                    #continue
                #filename = os.path.join(args.output_dir,
                #                        Path(item['filename']).name
                #                        ) if args.output_dir is not None else None
                gt_bboxes = item['gt_bboxes']
                gt_labels = item['gt_labels']
                # xmin ymin xmax ymax
                img = item['img']
                if contrastive_bool:
                    gt_bboxes = gt_bboxes.data.numpy()
                    gt_labels = gt_labels.data.numpy()
                    img = img.data.numpy()
                    img = np.moveaxis(img, 0, -1)
                else:
                    gt_bboxes = gt_bboxes.data.numpy()
                    gt_labels = gt_labels.data.numpy()
                    img = img.data.numpy()
                    img = np.moveaxis(img, 0, -1)
                    #img_tmp = img[:,:,0]
                    #img[:,:,0] = img[:,:,2]
                    #img[:,:,2] = img_tmp 
                
                imshow_det_bboxes(
                    img,
                    gt_bboxes,
                    gt_labels,
                    class_names=dataset.CLASSES,
                    show=not args.not_show,
                    wait_time=args.show_interval,
                    out_file=args.output_dir+str(lbl)+'_'+str(ssave_id)+'.jpg',
                    bbox_color=dataset.PALETTE,
                    text_color=(200, 200, 200),
                    mask_color=dataset.PALETTE)
                
                

            progress_bar.update()

def all_img():
    from PIL import Image
    np.random.seed(44)
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))
    stop = 0
    contrastive_bool = False
    
    for file_id, item_all in tqdm(enumerate(dataset)):
        if contrastive_bool:
            c_img = item_all['img_metas'].data
            item_three = [item_all['base'], item_all['novel'], item_all['augmented']]
            c_img = c_img['ori_filename']
        else:
            c_img = item_all
            name_img_meta = item_all['img_metas'].data
            c_img = name_img_meta['ori_filename']
            
            lbl = item_all['base']['gt_labels'].data
            if len(lbl.shape) > 1:
                lbl = lbl[0]
            else:
                lbl = lbl.item()
            if 'base' in item_all.keys():
                item_three = [item_all['base']]
            else:
                item_three = [item_all]
        ssave_id = 0
        stop += 1
        name = {0:'base', 1:'novel', 2:'aug'}
        for id, item in enumerate(item_three):
            gt_bboxes = item['gt_bboxes']
            gt_labels = item['gt_labels']
            # xmin ymin xmax ymax
            img = item['img']
            if contrastive_bool:
                gt_bboxes = gt_bboxes.data.numpy()
                gt_labels = gt_labels.data.numpy()
                img = img.data.numpy()
                img = np.moveaxis(img, 0, -1)
            else:
                gt_bboxes = gt_bboxes.data.numpy()
                gt_labels = gt_labels.data.numpy()
                img = img.data.numpy()
                img = np.moveaxis(img, 0, -1)
                #img_tmp = img[:,:,0]
                #img[:,:,0] = img[:,:,2]
                #img[:,:,2] = img_tmp 
            imshow_det_bboxes(
                img,
                gt_bboxes,
                gt_labels,
                class_names=dataset.CLASSES,
                show=not args.not_show,
                wait_time=args.show_interval,
                out_file=args.output_dir+str(stop)+'.jpg',
                bbox_color=dataset.PALETTE,
                text_color=(200, 200, 200),
                mask_color=dataset.PALETTE)   

            progress_bar.update()


if __name__ == '__main__':
    all_img()
