_base_ = [
    '../../_base_/datasets/tfa_dior.py',
    '../../_base_/schedules/schedule_1x_iter.py',
    '../tfa_r50_unfreeze_fpn.py',
    '../../_base_/default_runtime.py'
]

train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.1,
                min_pos_iou=0.1
            )
        )
)
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotDiorDefaultDataset',
        ann_dif = 'easy',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_1SHOT')],
        num_novel_shots=1,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))


evaluation = dict(
    interval=500,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
optimizer = dict(lr=0.001)

lr_config = dict(
    warmup=None, step=[
        500,
    ])
runner = dict(max_iters=2000)
checkpoint_config = dict(interval=2000)
# base model needs to be initialized with following script:
#   python -m tools.misc.initialize_bbox_head --src1 work_dirs/tfa_rsp_faster_rcnn_dior-base/latest.pth --method random_init --save-dir work_dirs/tfa_rsp_faster_rcnn_dior-base --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/tfa_rsp_faster_rcnn_dior-base/base_model_random_init_bbox_head.pth')
