_base_ = [
    '../../_base_/datasets/tfa_dior.py',
    '../../_base_/schedules/schedule_1x_iter.py',
    '../tfa_r50_all_unfreeze_fpn_DeFRCN.py',
    '../../_base_/default_shot_runtime.py'
]

# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='FewShotDiorDefaultDataset',
        ann_dif='hard',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_10SHOT')],
        num_novel_shots=10,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=1000, #2000,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
optimizer = dict(lr=0.001)
lr_config = dict(
    policy='step',
    warmup=None,
    step=[1000])#2000])
runner = dict(type='IterBasedRunner', max_iters=2000)#4000)
checkpoint_config = dict(interval=1999)#4000)
#runner = dict(type='IterBasedRunner', max_iters=6000)
#checkpoint_config = dict(interval=6000)

# base model needs to be initialized with following script:
#   python -m tools.misc.initialize_bbox_head --src1 work_dirs/DeFRCN_rsp_frcnn_dior_split1-base/latest.pth --method random_init --save-dir work_dirs/DeFRCN_rsp_frcnn_dior_split1-base --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/DeFRCN_rsp_frcnn_dior_split1-base/base_model_random_init_bbox_head.pth')
