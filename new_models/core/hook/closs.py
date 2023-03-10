from mmcv.parallel import is_module_wrapper

from mmcv.runner import HOOKS, Hook, Runner
from typing import Sequence

@HOOKS.register_module()
class ContrastiveLossWeight(Hook):
    """Unfreeze backbone network Hook.

    Args:
        change_weight_epoch (int): The epoch changing the loss weight
    """

    def __init__(self, 
        change_weight_epoch_cls=8, 
        change_weight_epoch_bbox=2, 
        increase_step_cls = .2,
        increase_step_bbox = .2, 
        number_of_epoch_step_bbox = 6,
        number_of_epoch_step_cls = 6, 
        change_weight_epoch_cosine=None, 
        increase_step_cosine = None,
        number_of_epoch_step_cosine = None):

        self.change_weight_epoch_cls = change_weight_epoch_cls
        self.increase_step_cls = increase_step_cls
        self.change_weight_epoch_bbox = change_weight_epoch_bbox
        self.increase_step_bbox = increase_step_bbox
        self.number_of_epoch_step_cls = number_of_epoch_step_cls
        self.number_of_epoch_step_bbox = number_of_epoch_step_bbox

        self.change_weight_epoch_cosine = change_weight_epoch_cosine
        self.increase_step_cosine = increase_step_cosine
        self.number_of_epoch_step_cosine = number_of_epoch_step_cosine

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        print(f'current epoch: {runner.epoch}')
        print(f'chaning at {self.change_weight_epoch_cls} for cls and at {self.change_weight_epoch_bbox} for bbox')
        if runner.epoch == self.change_weight_epoch_cls:
            print('change of cls loss weight')
            model = runner.model
            self.change_weight_epoch_cls += self.number_of_epoch_step_cls 
            if is_module_wrapper(model):
                model = model.module

            current_cls_weight = model.c_roi_head.bbox_head.loss_cls.loss_weight
            new_cls_value = min(self.increase_step_cls + current_cls_weight, 1.0)
            model.c_roi_head.bbox_head.loss_cls.loss_weight = new_cls_value
            print(f'new weight: {new_cls_value}, next epoch change: {self.change_weight_epoch_cls}')

        if runner.epoch == self.change_weight_epoch_bbox:
            print('change of bbox loss weight')
            model = runner.model
            self.change_weight_epoch_bbox += self.number_of_epoch_step_bbox 
            if is_module_wrapper(model):
                model = model.module

            current_bbox_weight = model.c_roi_head.bbox_head.loss_bbox.loss_weight
            new_bbox_value = min(self.increase_step_bbox + current_bbox_weight, 1.0)
            model.c_roi_head.bbox_head.loss_bbox.loss_weight = new_bbox_value
            print(f'new weight: {new_bbox_value}, next epoch change: {self.change_weight_epoch_bbox}')
        if self.change_weight_epoch_cosine is not None:
            if runner.epoch == self.change_weight_epoch_cosine:
                print('change of cosine loss weight')
                model = runner.model
                self.change_weight_epoch_cosine += self.number_of_epoch_step_cosine
                if is_module_wrapper(model):
                    model = model.module
                    
                current_cosine_weight = model.c_roi_head.bbox_head.loss_cosine.loss_weight
                new_cosine_value = min(self.increase_step_cosine + current_cosine_weight, 1.0)
                model.c_roi_head.bbox_head.loss_cosine.loss_weight = new_cosine_value
                print(f'new weight: {new_cosine_value}, next epoch change: {self.change_weight_epoch_cosine}')

@HOOKS.register_module()
class ContrastiveLossWeightBranch(Hook):
    """Unfreeze backbone network Hook.

    Args:
        change_weight_epoch (int): The epoch changing the loss weight
    """

    def __init__(self, 
        change_weight_epoch_cls=8, 
        change_weight_epoch_bbox=2, 
        increase_step_cls = .2,
        increase_step_bbox = .2, 
        number_of_epoch_step_bbox = 6,
        number_of_epoch_step_cls = 6, 
        change_weight_epoch_cosine=None, 
        increase_step_cosine = None,
        number_of_epoch_step_cosine = None):

        self.change_weight_epoch_cls = change_weight_epoch_cls
        self.increase_step_cls = increase_step_cls
        self.change_weight_epoch_bbox = change_weight_epoch_bbox
        self.increase_step_bbox = increase_step_bbox
        self.number_of_epoch_step_cls = number_of_epoch_step_cls
        self.number_of_epoch_step_bbox = number_of_epoch_step_bbox

        self.change_weight_epoch_cosine = change_weight_epoch_cosine
        self.increase_step_cosine = increase_step_cosine
        self.number_of_epoch_step_cosine = number_of_epoch_step_cosine

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        print(f'current epoch: {runner.epoch}')
        print(f'chaning at {self.change_weight_epoch_cls} for cls and at {self.change_weight_epoch_bbox} for bbox')
        if runner.epoch == self.change_weight_epoch_cls:
            print('change of cls loss weight')
            model = runner.model
            self.change_weight_epoch_cls += self.number_of_epoch_step_cls 
            if is_module_wrapper(model):
                model = model.module
            if model.roi_head.bbox_head.loss_cls is not None:
                current_cls_weight = model.roi_head.bbox_head.loss_cls.loss_weight
                new_cls_value = min(self.increase_step_cls + current_cls_weight, 1.0)
                model.roi_head.bbox_head.loss_cls.loss_weight = new_cls_value
                print(f'new weight: {new_cls_value}, next epoch change: {self.change_weight_epoch_cls}')

        if runner.epoch == self.change_weight_epoch_bbox:
            print('change of bbox loss weight')
            model = runner.model
            self.change_weight_epoch_bbox += self.number_of_epoch_step_bbox 
            if is_module_wrapper(model):
                model = model.module

            if model.roi_head.bbox_head.loss_bbox is not None:
                current_bbox_weight = model.roi_head.bbox_head.loss_bbox.loss_weight
                new_bbox_value = min(self.increase_step_bbox + current_bbox_weight, 1.0)
                model.roi_head.bbox_head.loss_bbox.loss_weight = new_bbox_value
                print(f'new weight: {new_bbox_value}, next epoch change: {self.change_weight_epoch_bbox}')
        if self.change_weight_epoch_cosine is not None:
            if runner.epoch == self.change_weight_epoch_cosine:
                print('change of cosine loss weight')
                model = runner.model
                self.change_weight_epoch_cosine += self.number_of_epoch_step_cosine
                if is_module_wrapper(model):
                    model = model.module
                if model.roi_head.bbox_head.loss_cosine is not None:
                    current_cosine_weight = model.roi_head.bbox_head.loss_cosine.loss_weight
                    new_cosine_value = min(self.increase_step_cosine + current_cosine_weight, 1.0)
                    model.roi_head.bbox_head.loss_cosine.loss_weight = new_cosine_value
                    print(f'new weight: {new_cosine_value}, next epoch change: {self.change_weight_epoch_cosine}')


@HOOKS.register_module()
class ContrastiveLossWeight_iter(Hook):
    """Unfreeze backbone network Hook.

    Args:
        change_weight_epoch (int): The epoch changing the loss weight
    """

    def __init__(self, 
        change_weight_epoch_cls=8, 
        change_weight_epoch_bbox=2, 
        increase_step_cls = .2,
        increase_step_bbox = .2, 
        number_of_epoch_step_bbox = 6,
        number_of_epoch_step_cls = 6, 
        change_weight_epoch_cosine=None, 
        increase_step_cosine = None,
        number_of_epoch_step_cosine = None):

        self.change_weight_epoch_cls = change_weight_epoch_cls
        self.increase_step_cls = increase_step_cls
        self.change_weight_epoch_bbox = change_weight_epoch_bbox
        self.increase_step_bbox = increase_step_bbox
        self.number_of_epoch_step_cls = number_of_epoch_step_cls
        self.number_of_epoch_step_bbox = number_of_epoch_step_bbox

        self.change_weight_epoch_cosine = change_weight_epoch_cosine
        self.increase_step_cosine = increase_step_cosine
        self.number_of_epoch_step_cosine = number_of_epoch_step_cosine

    def before_train_epoch(self, runner):
        # Unfreeze the backbone network.
        # Only valid for resnet.
        print(f'current epoch: {runner.epoch}')
        print(f'chaning at {self.change_weight_epoch_cls} for cls and at {self.change_weight_epoch_bbox} for bbox')
        if runner.iter == self.change_weight_epoch_cls:
            print('change of cls loss weight')
            model = runner.model
            self.change_weight_epoch_cls += self.number_of_epoch_step_cls 
            if is_module_wrapper(model):
                model = model.module

            current_cls_weight = model.c_roi_head.bbox_head.loss_cls.loss_weight
            new_cls_value = min(self.increase_step_cls + current_cls_weight, 1.0)
            model.c_roi_head.bbox_head.loss_cls.loss_weight = new_cls_value
            print(f'new weight: {new_cls_value}, next epoch change: {self.change_weight_epoch_cls}')

        if runner.iter == self.change_weight_epoch_bbox:
            print('change of bbox loss weight')
            model = runner.model
            self.change_weight_epoch_bbox += self.number_of_epoch_step_bbox 
            if is_module_wrapper(model):
                model = model.module

            current_bbox_weight = model.c_roi_head.bbox_head.loss_bbox.loss_weight
            new_bbox_value = min(self.increase_step_bbox + current_bbox_weight, 1.0)
            model.c_roi_head.bbox_head.loss_bbox.loss_weight = new_bbox_value
            print(f'new weight: {new_bbox_value}, next epoch change: {self.change_weight_epoch_bbox}')
        if self.change_weight_epoch_cosine is not None:
            if runner.iter == self.change_weight_epoch_cosine:
                print('change of cosine loss weight')
                model = runner.model
                self.change_weight_epoch_cosine += self.number_of_epoch_step_cosine
                if is_module_wrapper(model):
                    model = model.module
                    
                current_cosine_weight = model.c_roi_head.bbox_head.loss_cosine.loss_weight
                new_cosine_value = min(self.increase_step_cosine + current_cosine_weight, 1.0)
                model.c_roi_head.bbox_head.loss_cosine.loss_weight = new_cosine_value
                print(f'new weight: {new_cosine_value}, next epoch change: {self.change_weight_epoch_cosine}')
    
@HOOKS.register_module()
class ContrastiveLossDecayHook(Hook):
    """Hook for contrast loss weight decay used in FSCE.
    Args:
        decay_steps (list[int] | tuple[int]): Each item in the list is
            the step to decay the loss weight.
        decay_rate (float): Decay rate. Default: 0.5.
    """

    def __init__(self,
                 decay_steps: Sequence[int],
                 decay_rate: float = 0.5) -> None:
        assert isinstance(
            decay_steps,
            (list, tuple)), '`decay_steps` should be list or tuple.'
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def before_iter(self, runner: Runner) -> None:
        runner_iter = runner.iter + 1
        decay_rate = 1.0
        # update decay rate by number of iteration
        for step in self.decay_steps:
            if runner_iter > step:
                decay_rate *= self.decay_rate
        # set decay rate in the bbox_head
        if is_module_wrapper(runner.model):
            runner.model.module.roi_head.bbox_head.set_decay_rate(decay_rate)
        else:
            runner.model.roi_head.bbox_head.set_decay_rate(decay_rate)