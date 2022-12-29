_base_ = [
    '../../_base_/datasets/two_stage_dior.py',
    '../../_base_/schedules/schedule_few.py',
    '../mpsr_r50_fpn.py',
    '../../_base_/default_shot_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        dataset=dict(
            type='FewShotDiorDefaultDataset',
            ann_dif='hard',
            ann_cfg=[dict(method='MPSR', setting='SPLIT3_10SHOT')],
            num_novel_shots=10,
            num_base_shots=1,
            classes='ALL_CLASSES_SPLIT3')),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'))
    
evaluation = dict(
    interval=500, class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
checkpoint_config = dict(interval=2000)
optimizer = dict(
    lr=0.005,
    paramwise_cfg=dict(
        custom_keys=dict({'.bias': dict(lr_mult=2.0, decay_mult=0.0)})))
lr_config = dict(
    warmup_iters=500,
    warmup_ratio=1. / 3,
    step=[1300],
)
runner = dict(max_iters=2000)
# load_from = 'path of base training model'
load_from = (
    'work_dirs/mpsr_r50_base_split3/latest.pth')
model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(roi_layer=dict(aligned=False)),
        bbox_head=dict(init_cfg=[
            dict(
                type='Normal',
                override=dict(type='Normal', name='fc_cls', std=0.001))
        ])))