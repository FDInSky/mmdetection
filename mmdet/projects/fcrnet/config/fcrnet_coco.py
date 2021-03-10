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
            loss_weight=1.0),
        loss_iou=dict(
            type='CIoULoss', 
            loss_weight=1.0),
    ),
    refine_feats=[
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128],  
            # deform_groups=groups_num,
            deform_groups=1,
            use_score=True,
            repeat=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        ),
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128],  
            # deform_groups=groups_num,
            deform_groups=1,
            use_score=True,
            repeat=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg
        ),
    ],
    share_refine_feat=True,
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
                loss_weight=1.0),
            loss_iou=dict(
                type='CIoULoss',
                loss_weight=1.0),
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
                loss_weight=1.0),
            loss_iou=dict(
                type='CIoULoss',
                loss_weight=1.0),
        ),
    ],
    use_amp=False
)
# training and testing settings
train_cfg = dict(
    s0=dict(
        assigner=dict(
            type='FCRAssigner', 
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
                type='FCRAssigner', 
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
                type='FCRAssigner', 
                stage=1,
                iou_type=iou_type,
                iou_calculator=dict(type='BboxOverlaps2D'),
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
    ],
    stage_loss_weights=[1.5, 1.0, 0.5]
)
test_cfg = dict(
    s0=dict(
        nms_pre=1000,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=1000,
    ),
    sr=[
        dict(
            nms_pre=1000,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=100,
        ),
    ]
)
dataset_type = 'CocoDataset'
data_root = '/home/ai/data/zg_ping/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=False),
    dict(type='CutOut', n_holes=(2,16), cutout_shape=(8,8)),
    dict(type='MinIoURandomCrop', min_ious=(0.8, 0.9), min_crop_size=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.1, 1.5), saturation_range=(0.1, 1.5), hue_delta=18),
    dict(type='Resize', img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)], multiscale_mode='value', keep_ratio=True),
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
        img_scale=(1333, 800),
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
    samples_per_gpu=16,
    workers_per_gpu=16,
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
# Optim
total_epochs = 12
work_dir = './work_dirs/fcrnet_coco'
load_from = None
resume_from = None
nominal_batch_size = 32
gpus = 2
accumulate_interval = round(nominal_batch_size / (data['samples_per_gpu'] * gpus))
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.937, weight_decay=0.0005, nesterov=True,
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.)
)
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0.2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = dict(
#     type='AMPGradAccumulateOptimizerHook',
#     accumulation=accumulate_interval,
#     grad_clip=dict(max_norm=35, norm_type=2),
# )
custom_hooks = [
    dict(
        type='YoloV4WarmUpHook',
        warmup_iters=500,
        lr_weight_warmup=0.,
        lr_bias_warmup=0.1,
        momentum_warmup=0.9,
        priority='NORMAL'
    ),
    dict(
        type='YOLOV4EMAHook',
        momentum=0.9999,
        interval=accumulate_interval,
        warm_up=500 * accumulate_interval,
        resume_from=resume_from,
        priority='HIGH'
    )
]
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