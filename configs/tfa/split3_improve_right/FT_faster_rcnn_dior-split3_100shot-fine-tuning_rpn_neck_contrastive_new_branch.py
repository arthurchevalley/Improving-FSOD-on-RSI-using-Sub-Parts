_base_ = [
    '../new_branch_contrastive.py',
    '../../_base_/datasets/testing_norm_optimize_multi.py',
    '../../_base_/schedules/schedule_1x_iter.py',
    '../../_base_/default_shot_runtime.py'
]

# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.


model = dict(
    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
        )
    )
)

# base model needs to be initialized with following script:
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
checkpoint_config = dict(interval=30000)

# base model needs to be initialized with following script:
# python -m tools.misc.initialize_bbox_head --src1 work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_CONTRASTIVE_opti_BBOX/true_split3_extended.pth --method random_init --tar-name base_model_true_contrastive --save-dir work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_CONTRASTIVE_opti_BBOX --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/base_model_split3_random_init_bbox_head.pth')