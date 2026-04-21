_base_ = ['./ViTPose_small_simple_coco_256x192.py']

evaluation = dict(interval=1, metric='mAP', save_best='AP')
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

total_epochs = 1

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.8,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[1])

data_root = 'data/garbage_coco'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/images/',
    ),
    val=dict(
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/images/',
    ),
    test=dict(
        ann_file=f'{data_root}/annotations/person_keypoints_test2017.json',
        img_prefix=f'{data_root}/images/',
    ),
)
