_base_ = [
    '../../_base_/datasets/tfa_dior.py',
    '../../_base_/schedules/schedule_1x_iter.py',
    '../tfa_r50_unfreeze_fpn.py',
    '../../_base_/default_shot_runtime.py'
]


model = dict(
    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
 
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0)
        )
    )
)
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotDiorDefaultDataset',
        ann_dif='hard',
        ann_cfg=[dict(method='TFA', setting='SPLIT2_20SHOT')],
        num_novel_shots=20,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT2'),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=1000,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
optimizer = dict(lr=0.001)
lr_config = dict(
    policy='step',
    warmup=None,
    step=[4000, 4500])
runner = dict(type='IterBasedRunner', max_iters=5000)
checkpoint_config = dict(interval=5000)

# base model needs to be initialized with following script:
#   python -m tools.misc.initialize_bbox_head --src1 work_dirs/tfa_rsp_faster_rcnn_dior-base-split2/latest.pth --method random_init --save-dir work_dirs/tfa_rsp_faster_rcnn_dior-base-split2 --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/tfa_rsp_faster_rcnn_dior-base-split2/base_model_random_init_bbox_head.pth')
