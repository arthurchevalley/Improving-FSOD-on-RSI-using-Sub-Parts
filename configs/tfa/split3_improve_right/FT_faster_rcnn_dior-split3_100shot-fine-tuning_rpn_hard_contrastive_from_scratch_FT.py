_base_ = [
    '../../_base_/datasets/testing_norm_optimize_multi.py',
    '../../_base_/schedules/schedule_1x_iter.py',
    '../tfa_r50_unfreeze_fpn_contrastive_ft.py',
    '../../_base_/default_shot_runtime.py'
]

# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
number_of_class = 20
model = dict(
    c_rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    c_roi_head=dict(
        bbox_head=dict(
            num_classes=number_of_class,
            cosine_loss = dict(
                type='CosineSim',
                margin = .1,
                max_contrastive_loss = 3., 
                loss_weight=.01), #0.01
            loss_cls=dict(
                type='ConstellationLossCLS', 
                K=2, 
                max_contrastive_loss = 2, 
                loss_weight=.01#0.01
            ),
            loss_bbox=dict(
                type='ConstellationLossBBOX', 
                beta = 1.0/9.0,
                K=2, 
                max_contrastive_loss = 2, 
                loss_weight=.05
            )
        ),
    )
)

data = dict(
    train=dict(
        type='ContrastiveFewShotDiorDefaultDataset',
        ann_dif='hard',
        ann_cfg=[dict(method='TFA', setting='SPLIT3_100SHOT')],
        num_novel_shots=100,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT3'),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'))

custom_hooks = [dict(type='NumClassCheckHook'),
            dict(type='ContrastiveLossWeight_iter',
            change_weight_epoch_cls=10000, # 3
            change_weight_epoch_bbox=10000, # 2
            increase_step_cls = .1,
            increase_step_bbox = .05 ,
            number_of_epoch_step_bbox = 10001, # 2
            number_of_epoch_step_cls = 10001,
            change_weight_epoch_cosine = 10000,
            increase_step_cosine = .1,
            number_of_epoch_step_cosine = 10001) # 
            ]
evaluation = dict(
    interval=5000,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
optimizer = dict(lr=0.001)
lr_config = dict(
    policy='step',
    warmup=None,
    step=[20000, 25000])
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(interval=10000)

# base model needs to be initialized with following script:
# python -m tools.misc.initialize_bbox_head --src1 work_dirs/fsce_r50_fpn_split3_base/latest.pth --method random_init --tar-name base_model_split3 --save-dir work_dirs --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/base_model_split3_random_init_bbox_head.pth')
