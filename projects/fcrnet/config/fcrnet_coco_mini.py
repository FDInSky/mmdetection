# global settings
classes_num = 23
groups_num = 32
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=groups_num, requires_grad=True)
iou_type = 'iou'
# model settings
model = dict(
    type='FCRNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        # conv_cfg=conv_cfg,
        # norm_cfg=norm_cfg
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg
    ),
    bbox_head=dict(
        type='FCRBinaryHead',
        # num_classes=classes_num,
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.0,
            alpha=0.5,
            loss_weight=1.0),
        loss_iou=dict(
            type='CIoULoss', 
            loss_weight=2.0),
    ),
    refine_feats=[
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128],  
            # deform_groups=groups_num,
            deform_groups=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        ),
        # dict(
        #     in_channels=256,
        #     featmap_strides=[8, 16, 32, 64, 128],  
        #     # deform_groups=groups_num,
        #     deform_groups=1,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg
        # ),
    ],
    refine_heads=[
        dict(
            type='FCRRefineHead',
            num_classes=classes_num,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=dict(
                type='PseudoAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=(.0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=1.0,
                alpha=0.5,
                loss_weight=1.0),
            loss_iou=dict(
                type='CIoULoss',
                loss_weight=2.0),
        ),
        dict(
            type='FCRRefineHead',
            num_classes=classes_num,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=dict(
                type='PseudoAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=(.0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=1.0,
                alpha=0.5,
                loss_weight=1.0),
            loss_iou=dict(
                type='CIoULoss',
                loss_weight=1.0),
        ),
    ]
)
# training and testing settings
train_cfg = dict(
    s0=dict(
        assigner=dict(
            type='ATSSAssignerV2', 
            stage=0,
            iou_type=iou_type,
            iou_calculator=dict(type='BboxOverlaps2D'),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    sr=[
        dict(
            assigner=dict(
                type='ATSSAssignerV2', 
                stage=1,
                iou_type=iou_type,
                iou_calculator=dict(type='BboxOverlaps2D'),
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        dict(
            assigner=dict(
                type='ATSSAssignerV2', 
                stage=1,
                iou_type=iou_type,
                iou_calculator=dict(type='BboxOverlaps2D'),
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
    ],
    stage_loss_weights=[1.0, 1.0, 1.0]
)
test_cfg = dict(
    s0=dict(
        nms_pre=500,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=500,
    ),
    sr=[
        dict(
            nms_pre=500,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=100,
        ),
    ]
)