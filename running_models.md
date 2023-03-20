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
# Few-shot fine-tuning

For the few-shot fine-tuning, the few-shot dataset, number of shots and split must be defined. This is achieved by defining the data such as:
```shell
data = dict(
    train=dict(
        type='ContrastiveFewShotDiorDefaultDataset',
        ann_dif='hard',
        ann_cfg=[dict(method='TFA', setting='SPLIT2_10SHOT')],
        num_novel_shots=10,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT2'),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))
```
In additon, the fully-connected layers must be changed to match the number of classes. This is achieved by using initialize_bbox_head.py
```shell
python -m tools.initialize_bbox_head --src1 base_training_weight --method init_method --tar-name prefix_saving_name --save-dir saving_folder_of_new_weights --dior
```
Where _base_training_weight_ is the path to the base training weight, _init_method_ is how to initialise the classification weights, e.g. tests have been conducted with random_init, the new weights will be saved in folder _saving_folder_of_new_weights_ with the name starting with _prefix_saving_name_ and followed by _random_init_bbox_head.pth_ if random_init has been chosen. The last option --dior is needed to specify the dataset used.

If one want to load a queue from base training in fine-tuning, the queue must be initiliased. This is achieved by running:
```shell
python -m tools.misc.initialize_pretrain --src1 base_training_weight --save_dir saving_folder --queue class/random --nobg --base_withbg
```
Where _base_training_weight_ is the path to the base training weight, 'saving_folder' is the folder to save the queue features, 'queue' defines if the desired queue is a queue per class or random queue. Finally, if '--nobg' is added, the queue will not include background whereas not adding it provides a queue with background. 'nbr_base_class' and 'nbr_ft_class' can be added to specify the number of classes. By default, it is defined to match DIOR dataset with 15 base classes and 20 classes in total for fine-tuning. Finally, 'target_queue_length' can be added to specify the deisred queue length, by default 126. If the base queue is too small, the extra elements are initialised to None.


Finally, to load those newly changed weights, they must be loaded using the following command:
```shell
load_from = ('work_dirs/base_contrastive_perclass_1_bbox_model_split2_random_init_bbox_head.pth')
```

# Testing

To test model run the following:
```shell
python tools/test.py config_file work_dirs/weight_file --work-dir test_results_folder/ --out output_name.pkl --gpu-id 0 --eval mAP
```
