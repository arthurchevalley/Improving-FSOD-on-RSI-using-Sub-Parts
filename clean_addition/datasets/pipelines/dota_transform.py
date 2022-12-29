from ..builder import PIPELINES
import numpy as np
import cv2
from PIL import Image
import torch

@PIPELINES.register_module()
class FliterEmpty:

    def __call__(self, results):
        num_objs = 0
        for k in ['gt_bboxes', 'gt_masks', 'gt_labels']:
            if k in results:
                num_objs += len(results[k])
        if num_objs == 0:
            return None

        return results

@PIPELINES.register_module()
class LoadDOTASpecialInfo(object):

    def __init__(self,
                 img_keys=dict(gsd='gsd'),
                 ann_keys=dict(diffs='diffs', trunc='trunc'),
                 split_keys=dict(
                     split_sizes='sizes',
                     split_rates='rates',
                     split_gaps='gaps')):
        self.img_keys = img_keys
        self.ann_keys = ann_keys
        self.split_keys = split_keys

    def __call__(self, results):
        for res_key, img_key in self.img_keys.items():
            results[res_key] = results['img_info'][img_key]
        for res_key, split_key in self.split_keys.items():
            results[res_key] = results['split_info'][split_key]
        results['aligned_fields'] = []
        for res_key, ann_key in self.ann_keys.items():
            results[res_key] = results['ann_info'][ann_key]
            results['aligned_fields'].append(res_key)
        return results

@PIPELINES.register_module()
class DOTASpecialIgnore(object):

    def __init__(self,
                 ignore_diff=False,
                 ignore_truncated=False,
                 ignore_size=None,
                 ignore_real_scales=None):
        self.ignore_diff = ignore_diff
        self.ignore_truncated = ignore_truncated
        self.ignore_size = ignore_size
        self.ignore_real_scales = ignore_real_scales

    def __call__(self, results):
        for k in ['gt_bboxes', 'gt_masks', 'gt_labels']:
            if k in results:
                num_objs = len(results[k])
                break
        else:
            return results

        ignore = np.zeros((num_objs, ), dtype=np.bool)
        if self.ignore_diff:
            assert 'diffs' in results
            diffs = results['diffs']
            ignore[diffs == 1] = True

        if self.ignore_truncated:
            assert 'trunc' in results
            trunc = results['trunc']
            ignore[trunc == 1] = True

        if self.ignore_size:
            bboxes = results['gt_bboxes']
            wh = bboxes[:, 2:] - bboxes[:, :2]
            ignore[np.min(wh, axis=1) < self.ignore_size] = True

        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            gt_bboxes = bboxes[~ignore]
            gt_bboxes_ignore = bboxes[ignore]

            results['gt_bboxes'] = gt_bboxes
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            if 'gt_bboxes_ignore' not in results['bbox_fields']:
                results['bbox_fields'].append('gt_bboxes_ignore')

        if 'gt_labels' in results:
            results['gt_labels'] = results['gt_labels'][~ignore]

        for k in results.get('aligned_fields', []):
            results[k] = results[k][~ignore]

        return results


@PIPELINES.register_module()
class SaveTransformed(object):

    def __init__(self,
                 save_directory='/home/archeval/FSOD_remote/ds_browse/',
                 id_init = 0):
        self.directory = save_directory
        self.id = id_init
    def __call__(self, results):
        if self.directory is not None:
            #im = Image.open(im_tif)
            img = results['img']
            gt_bboxes = results['gt_bboxes']

            gt_lables = results['gt_labels']
            save_image = self.directory+str(self.id)+'.png'
            save_gt = self.directory+str(self.id)+'.txt'
            self.id += 1
            yes = (img._data).numpy()
            
            
            

            
        
            yes = np.moveaxis(yes, 0,-1)
            mean = - np.float64(np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, -1))
            stdinv = np.float64(np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, -1))
            
            a = cv2.multiply(stdinv, yes)  # inplace
            b = cv2.subtract(a, mean)  # inplace
            
            im = Image.fromarray(b,mode="RGB")
            
            im.save(save_image)
            np.savetxt(save_gt, (gt_bboxes._data).numpy())

            