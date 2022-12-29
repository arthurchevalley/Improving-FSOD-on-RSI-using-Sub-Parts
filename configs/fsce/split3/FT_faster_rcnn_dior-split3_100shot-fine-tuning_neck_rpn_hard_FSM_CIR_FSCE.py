
    
# base model needs to be initialized with following script:
# python -m tools.misc.initialize_bbox_head --src1 work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_CONTRASTIVE_opti_BBOX_CIR/latest.pth --method random_init --tar-name base_model_contrastive --save-dir work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_CONTRASTIVE_opti_BBOX_CIR --dior
# please refer to configs/detection/tfa/README.md for more details.

_base_ = [
    '../../_base_/datasets/tfa_dior.py',
    '../../_base_/schedules/schedule_1x_iter.py',
    '../fsce_r50_fpn_contrastive_CIR.py',
    '../../_base_/default_shot_runtime.py'
]

# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotDiorDefaultDataset',
        ann_dif='hard',
        ann_cfg=[dict(method='FSCE', setting='SPLIT3_100SHOT')],
        num_novel_shots=100,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT3'),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'))
evaluation = dict(
    interval=7500,
    class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
checkpoint_config = dict(interval=7500)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.5, step=[13000, 26000])
runner = dict(max_iters=30000)
custom_hooks = [
    dict(
        type='ContrastiveLossDecayHook',
        decay_steps=(6000, 10000),
        decay_rate=0.5)
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            with_weight_decay=True,
            loss_contrast=dict(iou_threshold=0.8, loss_weight=0.5))))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.

load_from = ('work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_FSM_CIR/base_model_CIR_random_init_bbox_head.pth')
