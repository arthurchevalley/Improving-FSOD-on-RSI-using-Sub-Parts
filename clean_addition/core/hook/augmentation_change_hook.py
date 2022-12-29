from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


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
            