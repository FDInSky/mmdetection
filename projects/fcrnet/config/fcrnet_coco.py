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
dataset_type = 'CocoDataset'
data_root = '/home/ai/data/zhangang2/ping_all_new/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=False),
    dict(type='CutOut', n_holes=(2,16), cutout_shape=(8,8)),
    dict(type='MinIoURandomCrop', min_ious=(0.8, 0.9), min_crop_size=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.1, 1.5), saturation_range=(0.1, 1.5), hue_delta=18),
    dict(type='Resize', img_scale=(640, 320), multiscale_mode='value', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg', 'pad_shape', 'scale_factor')),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/eval.json',
        img_prefix=data_root + 'images/eval/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/eval.json',
        img_prefix=data_root + 'images/eval/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[80, 110] 
)
total_epochs = 120
# runtime settings
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),]
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
evaluation = dict(interval=1, metric=['bbox'])
work_dir = './work_dirs/zg_ping_c23_fcrnet'
load_from = None #work_dir+"/epoch_94.pth"
resume_from = None #work_dir+"/epoch_37.pth"