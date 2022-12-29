_base_ = [
    '../../_base_/models/contrastive_r50_more.py', 
    '../../_base_/datasets/testing_norm_optimize_multi.py', 
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]


# model settings
number_of_class = 17
number_of_contrastive_class = 3
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
        type='CosContRoIHead_Branch',
        bbox_head=dict(
            type='AugContrastiveBBoxHead_Branch',
            mlp_head_channels=128,
            loss_bbox=dict(type='SmoothL1Loss_analyse', beta=1.0, loss_weight=1.0),
            loss_cosine = None,
            loss_c_cls = dict(
                type='CosineSim',
                margin = .5,
                max_contrastive_loss = 5., 
                loss_weight=1.),
            loss_c_bbox=dict(
                type='ConstellationLossBBOX', 
                K=2, 
                max_contrastive_loss = 3., 
                loss_weight=.05
            ),
            scale=20,
            num_classes=number_of_class,
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
            ]
        )
    )
    
)


log_config = dict(
    interval=1)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        classes='BASE_CLASSES_SPLIT3'),
    val=dict(
            classes='BASE_CLASSES_SPLIT3'),
    test=dict(
            classes='BASE_CLASSES_SPLIT3'))
evaluation = dict(
    interval=4,
    metric='mAP',
    class_splits=['BASE_CLASSES_SPLIT3'])

custom_hooks = [dict(type='NumClassCheckHook'),
                dict(type='ContrastiveLossWeightBranch',
                change_weight_epoch_cls=2, # 3
                change_weight_epoch_bbox=2, # 2
                increase_step_cls = .05,
                increase_step_bbox = .05 ,
                number_of_epoch_step_bbox = 2, # 2
                number_of_epoch_step_cls = 2) # 
                ]
# first ch. = 2 et nbr = 2
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
    # yapf:disable

auto_scale_lr = dict(enable=False, base_batch_size=2)
checkpoint_config = dict(interval=3)

lr_config = dict(
    policy='step',
    warmup_iters=1,
    step=[8, 11])
load_from = ('work_dirs/tfa_rsp_faster_rcnn_dior-base_split3_CONTRASTIVE_redo/init_e3.pth')

