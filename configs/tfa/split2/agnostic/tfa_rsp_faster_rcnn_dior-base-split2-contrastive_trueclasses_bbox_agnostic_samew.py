_base_ = [
    '../../../_base_/models/contrastive_r50_noRoI.py',
    '../../../_base_/datasets/testing_norm_optimize_multi_rotate_base_s2.py',
    '../../../_base_/schedules/schedule_1x.py',
    '../../../_base_/default_runtime.py'
]



# model settings

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='rsp-resnet-50-ckpt_ready.pth',
            )
    ),
    rpn_head=dict(
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta = 1.0/9.0,loss_weight=1.0)
    ),
    roi_head=dict(
        type='Agnostic_DualCosContRoIHead_Branch_novcls_nobbox_separate',
        bbox_head=dict(
            type='Agnostic_QueueAugContrastiveBBoxHead_Branch',
            mlp_head_channels=128,
            with_weight_decay=True,
            main_training=True,
            same_class = True,
            same_class_all = True,
            reg_class_agnostic=True, 
            with_cls = False,
            num_classes=15,

            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0), #CIoULoss
            loss_cosine=dict(
                type='QueueDualSupervisedContrastiveLoss_light',
                temperature=0.2,
                iou_threshold=0.5,
                loss_weight=0.5,
                reweight_type='none',
                no_bg = True),
            loss_c_cls=dict(
                type='QueueDualSupervisedContrastiveLoss_light',
                temperature=0.2,
                iou_threshold=0.5,
                loss_weight=0.5,
                reweight_type='none',
                no_bg = True),
            loss_base_aug=dict(
                type='QueueDualSupervisedContrastiveLoss_light',
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
            queue_length = 60
        )
    )
)


data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT2'),
    val=dict(classes='BASE_CLASSES_SPLIT2'),
    test=dict(classes='BASE_CLASSES_SPLIT2'))
evaluation = dict(
    interval=14,
    metric='mIoU',
    class_splits=['BASE_CLASSES_SPLIT2'])

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7, 10])



optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
auto_scale_lr = dict(enable=False, base_batch_size=2)
checkpoint_config = dict(interval=4)