_base_ = [
    '../../../_base_/datasets/tfa_dior.py',
    '../../../_base_/schedules/schedule_1x_iter.py',
    '../../tfa_r50_unfreeze_rpn.py',
    '../../../_base_/default_shot_runtime.py'
]


data = dict(
    train=dict(
        type='FewShotDiorDefaultDataset',
        dior_folder_path='/home/data/dior',
        ann_dif='hard',
        ann_cfg=[dict(method='TFA', setting='SPLIT2_5SHOT')],
        num_novel_shots=5,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT2'),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=500,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
optimizer = dict(lr=0.001)
lr_config = dict(
    policy='step',
    warmup=None,
    step=[1000, 1500])
runner = dict(type='IterBasedRunner', max_iters=2000)
checkpoint_config = dict(interval=2000)

# base model needs to be initialized with following script:
# python -m tools.initialize_bbox_head --src1 work_dirs/tfa_rsp_faster_rcnn_dior-base-split2/latest.pth --method random_init --tar-name base_model_split2 --save-dir work_dirs --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/FINAL_WEIGHTS/base_model_random_init_bbox_head_split2.pth')