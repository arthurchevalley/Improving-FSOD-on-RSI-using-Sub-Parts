_base_ = [
    '../../../new_branch_contrastive_loss_cls_region_nobbox_separate.py',
    '../../../../../_base_/datasets/testing_norm_optimize_multi_rotate_s2.py',
    '../../../../../_base_/schedules/schedule_1x_iter.py',
    '../../../../../_base_/default_shot_runtime.py'
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
            loss_cosine=dict(
                type='QueueDualSupervisedContrastiveLoss_class',
                temperature=0.2,
                iou_threshold=0.5,
                loss_weight=0.5,
                reweight_type='none',
                no_bg = True),
            loss_c_cls=dict(
                type='QueueDualSupervisedContrastiveLoss_class',
                temperature=0.2,
                iou_threshold=0.5,
                loss_weight=0.5,
                reweight_type='none',
                no_bg = True),
            loss_base_aug=dict(
                type='QueueDualSupervisedContrastiveLoss_class',
                temperature=0.2,
                iou_threshold=0.5,
                loss_weight=0.5,
                reweight_type='none',
                no_bg = True),
            loss_c_bbox=None,
            to_norm_cls = True,
            queue_path = None,
            use_queue = True,
            use_base_queue = True,
            use_novel_queue = True,
            use_aug_queue = True,
            queue_length = 120
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=2000),
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.4, neg_iou_thr=0.4, min_pos_iou=0.4),
            sampler=dict(num=256)))
)
# base model needs to be initialized with following script:
data = dict(
    train=dict(
        type='ContrastiveFewShotDiorDefaultDataset',
        ann_dif='hard',
        ann_cfg=[dict(method='TFA', setting='SPLIT2_10SHOT')],
        num_novel_shots=10,
        num_base_shots=10,
        classes='ALL_CLASSES_SPLIT2'),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'))
evaluation = dict(
    interval=2500,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
optimizer = dict(lr=0.001)
lr_config = dict(
    policy='step',
    warmup=None,
    step=[4000, 6000])
runner = dict(type='IterBasedRunner', max_iters=7500)
checkpoint_config = dict(interval=7500)

# base model needs to be initialized with following script:
# python -m tools.misc.initialize_bbox_head --src1 work_dirs/tfa_rsp_faster_rcnn_dior-base-split2-contrastive_trueclasses_bbox_agnostic/latest.pth --method random_init --tar-name contrastive_agnostic_model_split2 --save-dir work_dirs --dior
# please refer to configs/detection/tfa/README.md for more details.


load_from = ('work_dirs/contrastive_agnostic_model_split2_random_init_bbox_head.pth')