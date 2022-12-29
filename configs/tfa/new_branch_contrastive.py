_base_ = ['../_base_/models/contrastive_r50_more.py']

model = dict(
    frozen_parameters=[
        'backbone',
        'neck'
    ],
    backbone=dict(frozen_stages=4),
    roi_head=dict(
        type='CosContRoIHead_Branch',
        bbox_head=dict(
            type='AugContrastiveBBoxHead_Branch',
            mlp_head_channels=128,
            loss_cosine = dict(
                type='CosineSim',
                margin = .5,
                max_contrastive_loss = 5., 
                loss_weight=.5),
            loss_c_cls = dict(
                type='CosineSim',
                margin = .5,
                max_contrastive_loss = 5., 
                loss_weight=.5),
            scale=20,
            num_classes=20,
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
