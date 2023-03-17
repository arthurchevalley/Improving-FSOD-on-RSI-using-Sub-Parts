_base_ = [
    'new_branch_contrastive_loss_cls_region_nobbox_separate.py',
    '../../_base_/datasets/testing_norm_optimize_multi_rotate_s2_more_browse.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_shot_runtime.py'
]




model = dict(
    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
 
    roi_head=dict(
        bbox_head=dict(
            type='QueueAugContrastiveBBoxHead_Branch_classqueue_replace',
            num_classes=20,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=10.0),
            loss_cosine=None,
            loss_all=dict(
                type='QueueDualSupervisedContrastiveLoss_class_all',
                temperature=0.2,
                iou_threshold=.5,
                loss_weight=1.,
                reweight_type='none',
                no_bg = True),
            loss_c_cls=None,
            loss_base_aug=None,
            loss_c_bbox=None,
            to_norm_cls = True,
            queue_path = None,
            use_queue = True,
            use_base_queue = True,
            use_novel_queue = True,
            use_aug_queue = True
        )
    )
)

# base model needs to be initialized with following script:
data = dict(
    train=dict(
        type='ContrastiveFewShotDiorDefaultDataset',
        dior_folder_path='/home/archeval/mmdetection/CATNet/mmdetection/data/dior',
        ann_dif='hard',
        ann_cfg=[dict(method='TFA', setting='SPLIT2_10SHOT')],
        num_novel_shots=10,
        num_base_shots=1,
        classes='ALL_CLASSES_SPLIT2'),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))

runner = dict(type='EpochBasedRunner', max_epochs=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
# base model needs to be initialized with following script:
# python -m tools.misc.initialize_bbox_head --src1 work_dirs/tfa_rsp_faster_rcnn_dior-base-split2/latest.pth --method random_init --tar-name base_model_split2 --save-dir work_dirs --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/FINAL_WEIGHTS/tfa_rsp_faster_rcnn_dior-base-split2/base_model_random_init_bbox_head.pth')