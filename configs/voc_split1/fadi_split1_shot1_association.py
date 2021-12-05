_base_ = '../_base_/faster_rcnn_r50_caffe_fpn.py'

custom_imports = dict(
    imports=['fadi.models.cosine_sim_bbox_head', 'fadi.datasets.few_shot_voc'])

model = dict(pretrained='open-mmlab://detectron2/resnet101_caffe',
             backbone=dict(depth=101, frozen_stages=4),
             rpn_head=dict(anchor_generator=dict(scale_major=False)),
             roi_head=dict(bbox_head=dict(type='CosineSimBBoxHead',
                                          num_classes=15,
                                          loss_bbox=dict(loss_weight=0.))))

unfreeze_layers = ('roi_head.bbox_head.shared_fcs.1', )

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                    (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                    (1333, 736), (1333, 768), (1333, 800)],
         multiscale_mode='value',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

split = 1
shot = 1

dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotVOCDataset',
        ann_file=None,  # not used in few shot voc
        split=split,
        shot=shot,
        img_prefix=None,
        associate=True,
        pipeline=train_pipeline),
    val=dict(type='FewShotVOCTestDataset',
             ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
             split=split,
             img_prefix=data_root + 'VOC2007/',
             pipeline=test_pipeline,
             associate=True),
    test=dict(type='FewShotVOCTestDataset',
              ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
              split=split,
              img_prefix=data_root + 'VOC2007/',
              pipeline=test_pipeline,
              associate=True))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1,
                 warmup_ratio=0.001,
                 step=[3000])

# Runner type
runner = dict(type='IterBasedRunner', max_iters=4000)

checkpoint_config = dict(interval=1000)
evaluation = dict(interval=1000, metric='mAP')

load_from = 'models/voc_split1_base.pth'
