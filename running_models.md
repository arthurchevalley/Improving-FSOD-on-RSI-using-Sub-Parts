# Data Setup


# Base Training & Fine-Tunning
To run models, each of them must be defined in a file, generally stored in the confgis folder.
For this project, all training files are located in configs/tfa/split2/...

For illustration purposes, the configs/tfa/split2/test_env.py file is used.

The first part of the file defines the base component of the file.
```shell
_base_ = [
    '../../_base_/models/contrastive_r50_noRoI.py',
    '../../_base_/datasets/testing_norm_optimize_multi_rotate_base_s2.py',
    '../../_base_/schedules/schedule_1x_iter.py',
    '../../_base_/default_shot_runtime.py'
]
```
The first line defines the general model to use and the second the dataset pipeline. The third line defines the training length, optimizer and when saving or evaluating the model. Finally, the forth line defines the used hooks and other general setup.
Note that all of those are initialising the model but can be overloaded by the code following. It allows to have the similar base model and  specific parameters changes.

**Overwritting model settings**
To change specific component of the model, the desired parameter are set after. In this exemple, a pre-trained backbone is loaded from a weight file. Then the RPN head loss and the RoI head are changed. All 'type' parameters are defined in the 'new_models' folder.
```shell
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='rsp-resnet-50-ckpt_ready.pth',
            )
    ),
    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    roi_head=dict(
        type='DualCosContRoIHead_Branch_novcls_nobbox_separate',
        bbox_head=dict(
            type='QueueAugContrastiveBBoxHead_Branch_classqueue_replace_withbg',
            mlp_head_channels=128,
            with_weight_decay=True,
            main_training=True,
            same_class = True,
            same_class_all=True,
            num_classes=15,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=10.0),
            loss_cosine=dict(
                type='QueueDualSupervisedContrastiveLoss_class_nobg',
                temperature=0.2,
                iou_threshold=0.5,
                loss_weight=0.5,
                reweight_type='none',
                no_bg = True),
            loss_c_cls=dict(
                type='QueueDualSupervisedContrastiveLoss_class_nobg',
                temperature=0.2,
                iou_threshold=0.5,
                loss_weight=0.5,
                reweight_type='none',
                no_bg = True),
            loss_base_aug=dict(
                type='QueueDualSupervisedContrastiveLoss_class_nobg',
                temperature=0.2,
                iou_threshold=0.5,
                loss_weight=0.5,
                reweight_type='none',
                no_bg = True),
            loss_c_bbox=None,
            to_norm_cls = True,
            queue_path = 'no_class_features_base.p',
            use_queue = True,
            use_base_queue = True,
            use_novel_queue = True,
            use_aug_queue = True,
            queue_length = 128
        )
    )
)
```
**Overwritting training settings**
Custom hook, such a the decrease of regression weight during training, can be defined here. 
```shell
custom_hooks = [
     dict(
        type='BBOX_WEIGHT_CHANGE',
            change_weight_epoch_bbox=6, 
            increase_ratio_bbox = .3, 
            number_of_epoch_step_bbox = 3
    )
]
```

The data used for training can be defined here. The mentionned split are defined the in dataset folder present in 'new_models/datases'.
```shell
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT2'),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'))
```

cometML logging can be defined by adding the following lines in the file. Note that the 'personal-key-to-replace' must be replaced by the user api-key.
```shell
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='CometMLLoggerHook', 
            project_name='logger_comet_ml',
            api_key= 'personal-key-to-replace')
    ])
```