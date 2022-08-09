# dataset settings
dataset_type = 'CocoDataset'
# data_root = '/content/mmdetection/1A_data/' # dataset 위치
classes = ('fish','jellyfish','penguin','puffin','shark','starfish','stingray') ## class 정의
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True), ## image size 변경
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800), ## image size 변경
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
data = dict(
    samples_per_gpu=1, ## gpu당 batch사이즈 며층로 할건지 , 
    workers_per_gpu=2, ## data loader를 만들 때 worker개서 선언해주는 것과 동일
    train=dict(
        type=dataset_type,
        ann_file= '/home/ubuntu/test/mmdetection/configs/1A_data/train/_annotations.coco.json', ## train annotation file 위치
        img_prefix= '/home/ubuntu/test/mmdetection/configs/1A_data/train/', ## data root 위치
        classes = classes, ## classes 추가
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file= '/home/ubuntu/test/mmdetection/configs/1A_data/valid/_annotations.coco.json', ## val annotation file 위치
        img_prefix='/home/ubuntu/test/mmdetection/configs/1A_data/valid/', ## data root 위치
        classes = classes, ## classes 추가
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file= '/home/ubuntu/test/mmdetection/configs/1A_data/test/_annotations.coco.json', ## test annotation file 위치
        img_prefix='/home/ubuntu/test/mmdetection/configs/1A_data/test/', ## data root 위치
        classes = classes, ## classes 추가
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')