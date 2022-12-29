_base_ = [
    'new_branch_contrastive_loss.py',
    '../../_base_/datasets/testing_norm_optimize_multi_rotate.py',
    '../../_base_/schedules/schedule_1x_iter.py',
    '../../_base_/default_shot_runtime.py'
]

# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.


model = dict(
    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=0.0),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.0),
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=0.0),
            loss_cosine=dict(type='DualSupervisedContrastiveLoss', iou_threshold=0.8, loss_weight=0.0),
            loss_c_cls=dict(type='DualSupervisedContrastiveLoss', iou_threshold=0.8, loss_weight=0.0),
            loss_base_aug=dict(type='DualSupervisedContrastiveLoss', iou_threshold=0.8, loss_weight=0.0),
            loss_c_bbox=dict(type='IoULossBBOX', loss_weight=0.0),
            queue_path = 'init_queue_features.p'
        )
    )
)

# base model needs to be initialized with following script:
data = dict(
    train=dict(
        type='ContrastiveFewShotDiorDefaultDataset',
        ann_dif='hard',
        ann_cfg=[dict(method='TFA', setting='SPLIT3_1SHOT')],
        num_novel_shots=1,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT3'),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'))
evaluation = dict(
    interval=6250,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
checkpoint_config = dict(interval=6250)
optimizer = dict(lr=0.0)
runner = dict(max_iters=20)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='CometMLLoggerHook', 
        #    project_name='logger_comet_ml',
        #    api_key= 'UavGjAWatUgY4kp6T3tv3VWuS')
    ])
# base model needs to be initialized with following script:
# python -m tools.misc.initialize_bbox_head --src1 work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_CONTRASTIVE_opti_BBOX/true_split3_extended.pth --method random_init --tar-name base_model_true_contrastive --save-dir work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_CONTRASTIVE_opti_BBOX --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/base_model_split3_random_init_bbox_head.pth')