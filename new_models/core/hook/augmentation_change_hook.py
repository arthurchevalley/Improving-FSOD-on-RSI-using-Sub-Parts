
from mmcv.parallel import is_module_wrapper

from mmcv.runner import HOOKS, Hook, Runner
from typing import Sequence

@HOOKS.register_module()
class AugmentationChange(Hook):
    """Unfreeze backbone network Hook.

    Args:
        change_weight_epoch (int): The epoch changing the loss weight
    """

    def __init__(self, 
        change_augmentation_epoch_rotate=8, 
        change_augmentation_epoch_flip=2, 
        increase_step_rotate = .2,
        increase_step_flip = .2, 
        number_of_epoch_step = 6):

        self.change_augmentation_epoch_rotate = change_augmentation_epoch_rotate
        self.change_augmentation_epoch_flip = change_augmentation_epoch_flip

        self.increase_step_rotate = increase_step_rotate
        self.increase_step_flip = increase_step_flip

        self.number_of_epoch_step = number_of_epoch_step

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.

        if runner.epoch == self.change_augmentation_epoch_flip or runner.epoch == self.change_augmentation_epoch_rotate:
            print('change of aug prob')
            model = runner.model
            
            current_pipeline = runner.data_loader.dataset.multi_pipelines['augmented']
        
            if is_module_wrapper(current_pipeline):
                    current_pipeline = current_pipeline.module
            flip = current_pipeline.transforms

            if runner.epoch == self.change_augmentation_epoch_rotate:
                self.change_augmentation_epoch_rotate += self.number_of_epoch_step 
                flip[-2].prob += self.increase_step_rotate
                flip[-2].prob = min(flip[-2].prob, 1.0)
            if runner.epoch == self.change_augmentation_epoch_flip:   
                self.change_augmentation_epoch_flip += self.number_of_epoch_step      
                flip[-1].flip_ratio += self.increase_step_flip
                flip[-1].flip_ratio = min(flip[-1].flip_ratio, 1.0)
            

@HOOKS.register_module()
class BBOX_WEIGHT_CHANGE(Hook):
    """Unfreeze backbone network Hook.

    Args:
        change_weight_epoch (int): The epoch changing the loss weight
    """

    def __init__(self, 
        change_weight_epoch_bbox=7, 
        increase_ratio_bbox = .5, 
        number_of_epoch_step_bbox = 3):

        self.change_weight_epoch_bbox = change_weight_epoch_bbox
        self.increase_ratio_bbox = increase_ratio_bbox
        self.number_of_epoch_step_bbox = number_of_epoch_step_bbox

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        print(f'current epoch: {runner.epoch}')
        print(f'chaning at {self.change_weight_epoch_bbox} for bbox')

        if runner.epoch == self.change_weight_epoch_bbox:
            print('change of bbox loss weight')
            model = runner.model
            self.change_weight_epoch_bbox += self.number_of_epoch_step_bbox 
            if is_module_wrapper(model):
                model = model.module

            if model.roi_head.bbox_head.loss_bbox is not None:
                current_bbox_weight = model.roi_head.bbox_head.loss_bbox.loss_weight
                new_bbox_value = max(self.increase_ratio_bbox*current_bbox_weight, 1.0)
                model.roi_head.bbox_head.loss_bbox.loss_weight = new_bbox_value
                print(f'new weight: {new_bbox_value}, next epoch change: {self.change_weight_epoch_bbox}')

@HOOKS.register_module()
class Bbox_Cont_Weight_Change(Hook):
    """Unfreeze backbone network Hook.

    Args:
        change_weight_epoch (int): The epoch changing the loss weight
    """

    def __init__(self, 
        change_weight_epoch_bbox=7, 
        increase_ratio_bbox = .5, 
        number_of_epoch_step_bbox = 3,
        new_contrastive_loss = .5):

        self.change_weight_epoch_bbox = change_weight_epoch_bbox
        self.increase_ratio_bbox = increase_ratio_bbox
        self.number_of_epoch_step_bbox = number_of_epoch_step_bbox
        self.new_contrastive_loss = new_contrastive_loss

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        print(f'current epoch: {runner.epoch}')
        print(f'chaning at {self.change_weight_epoch_bbox} for bbox')

        if runner.epoch == self.change_weight_epoch_bbox:
            print('change of bbox loss weight')
            model = runner.model
            self.change_weight_epoch_bbox += self.number_of_epoch_step_bbox 
            if is_module_wrapper(model):
                model = model.module

            if model.roi_head.bbox_head.loss_bbox is not None:
                current_bbox_weight = model.roi_head.bbox_head.loss_bbox.loss_weight
                new_bbox_value = max(self.increase_ratio_bbox*current_bbox_weight, 1.0)
                model.roi_head.bbox_head.loss_bbox.loss_weight = new_bbox_value
                print(f'new weight: {new_bbox_value}, next epoch change: {self.change_weight_epoch_bbox}')
            if not model.roi_head.bbox_head.loss_cosine.loss_weight: 
                model.roi_head.bbox_head.loss_cosine.loss_weight = self.new_contrastive_loss

            if not model.roi_head.bbox_head.loss_c_cls.loss_weight: 
                model.roi_head.bbox_head.loss_c_cls.loss_weight = self.new_contrastive_loss

            if not model.roi_head.bbox_head.loss_base_aug.loss_weight: 
                model.roi_head.bbox_head.loss_base_aug.loss_weight = self.new_contrastive_loss
            
@HOOKS.register_module()
class save_bbox_feat(Hook):
    """Unfreeze backbone network Hook.

    Args:
        change_weight_epoch (int): The epoch changing the loss weight
    """

    def __init__(self, 
        save_epoch=11, 
        number_of_epoch_step = 3):

        self.save_epoch = save_epoch
        self.number_of_epoch_step = number_of_epoch_step

    def after_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        print(f'current epoch: {runner.epoch}')
        print(f'chaning at {self.save_epoch} for bbox')

        if runner.epoch == self.save_epoch:
            print('change of bbox loss weight')
            model = runner.model
            self.save_epoch += self.number_of_epoch_step 
            if is_module_wrapper(model):
                model = model.module
            model.roi_head.bbox_head.save_bbox_feat = True
            print(f'saving bbox feat True, next epoch change: {self.save_epoch}')


@HOOKS.register_module()
class save_bbox_feat_iter(Hook):
    """Unfreeze backbone network Hook.

    Args:
        change_weight_epoch (int): The epoch changing the loss weight
    """

    def __init__(self, 
        save_iter=100, 
        number_of_iter_step = 3):

        self.save_epoch = save_iter
        self.number_of_epoch_step = number_of_iter_step

    def before_train_iter(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        if runner.iter == 0:
            print(f'chaning at {self.save_epoch} for bbox')

        if runner.iter == self.save_epoch:
            print('change of bbox loss weight')
            model = runner.model
            self.save_epoch += self.number_of_epoch_step 
            if is_module_wrapper(model):
                model = model.module
            print(model.roi_head.bbox_head.save_bbox_feat)
            model.roi_head.bbox_head.save_bbox_feat = True
            print(f'saving bbox feat True, next epoch change: {self.save_epoch}')
    
