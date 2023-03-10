_base_ = ['../../_base_/models/contrastive_r50_noRoI.py']

model = dict(
    
    frozen_parameters=[
        'backbone',
    ],

    backbone=dict(frozen_stages=4),
    roi_head=dict(
        type='DualCosContRoIHead_Branch_novcls_nobbox_separate',
        bbox_head=dict(
            type='QueueAugContrastiveBBoxHead_Branch',
            mlp_head_channels=128,
            with_weight_decay=True,
            loss_cosine=dict(
                type='QueueDualSupervisedContrastiveLoss',
                temperature=0.2,
                iou_threshold=0.8,
                loss_weight=0.5,
                reweight_type='none'),
            loss_c_cls=dict(
                type='QueueDualSupervisedContrastiveLoss',
                temperature=0.2,
                iou_threshold=0.8,
                loss_weight=0.5,
                reweight_type='none'),
            loss_base_aug=dict(
                type='QueueDualSupervisedContrastiveLoss',
                temperature=0.2,
                iou_threshold=0.8,
                loss_weight=0.5,
                reweight_type='none'),
            loss_c_bbox=dict(
                type='IoULossBBOX',
                loss_weight=0.5),
            queue_path = 'init_queue.p',
            use_queue = False,
            num_classes=20,
            scale=20,
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001)),
                dict(
                    type='Caffe2Xavier',
                    override=dict(
                        type='Caffe2Xavier', name='contrastive_head'))
            ])),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=2000),
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.4, neg_iou_thr=0.4, min_pos_iou=0.4),
            sampler=dict(num=256)))
            )