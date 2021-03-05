# Model
model = dict(
    type='YOLOV4',
    backbone=dict(
        type='DarknetCSP',
        architecture='yolov5',
        scale='l5p',
        out_indices=[2, 3, 4]),
    neck=dict(
        type='PAFPNCSP',
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        csp_repetition=2),
    bbox_head=dict(
        type='YOLOV4Head',
        num_classes=80,
        in_channels=[256, 512, 1024],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(12, 16), (19, 36), (40, 28)],        # P3/8
                        [(36, 75), (76, 55), (72, 146)],       # P4/16
                        [(142, 110), (192, 243), (459, 401)]], # P5/32
            strides=[8, 16, 32]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[8, 16, 32],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(
            type='MSELoss', 
            loss_weight=2.0, 
            reduction='sum')
    ),
    use_amp=True
)
# training and testing settings
train_cfg=dict(
    assigner=dict(
        type='GridAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0
    )
)
test_cfg=dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=100
)
# Dataset
dataset_type = 'CocoDataset'
data_root = '/home/ai/data/coco/'
img_norm_cfg = dict(
    mean=[114, 114, 114], std=[255, 255, 255], to_rgb=True)
train_pipeline = [
    dict(type='MosaicPipeline',
         individual_pipeline=[
             dict(type='LoadImageFromFile'),
             dict(type='LoadAnnotations', with_bbox=True),
             dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
         ],
         pad_val=114),
    dict(type='Albu',
         update_pad_shape=True,
         skip_img_without_anno=False,
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             min_area=4,
             min_visibility=0.2,
             label_fields=['gt_labels'],
             check_each_transform=False
         ),
         transforms=[
             dict(type='PadIfNeeded',
                  min_height=1920,
                  min_width=1920,
                  border_mode=0,
                  value=(114, 114, 114),
                  always_apply=True),
             dict(type='RandomCrop',
                  width=1280,
                  height=1280,
                  always_apply=True),
             dict(
                 type='RandomScale',
                 scale_limit=0.5,
                 interpolation=1,
                 always_apply=True),
             dict(
                 type='CenterCrop',
                 width=640,
                 height=640,
                 always_apply=True),
             dict(
                 type='HorizontalFlip',
                 p=0.5)
         ]),
    dict(type='HueSaturationValueJitter',
         hue_ratio=0.015,
         saturation_ratio=0.7,
         value_ratio=0.4),
    dict(type='GtBBoxesFilter',
         min_size=2,
         max_aspect_ratio=20),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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
        pipeline=test_pipeline)
)
# Optim
nominal_batch_size = 64
gpus = 2
accumulate_interval = round(nominal_batch_size / (data['samples_per_gpu'] * gpus))
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.937, weight_decay=0.0005, nesterov=True,
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.)
)
optimizer_config = dict(
    type='AMPGradAccumulateOptimizerHook',
    accumulation=accumulate_interval,
    grad_clip=dict(max_norm=35, norm_type=2),
)
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0.2)
total_epochs = 300
load_from = None
resume_from = None
custom_hooks = [
    dict(
        type='YoloV4WarmUpHook',
        warmup_iters=10000,
        lr_weight_warmup=0.,
        lr_bias_warmup=0.1,
        momentum_warmup=0.9,
        priority='NORMAL'
    ),
    dict(
        type='YOLOV4EMAHook',
        momentum=0.9999,
        interval=accumulate_interval,
        warm_up=10000 * accumulate_interval,
        resume_from=resume_from,
        priority='HIGH'
    )
]
# Running
evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=5, max_keep_ckpts=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
cudnn_benchmark = True
