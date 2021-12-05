_base_ = 'fadi_split3_shot1_association.py'

split = 3
shot = 2
data = dict(train=dict(split=split, shot=shot),
            val=dict(split=split),
            test=dict(split=split))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1,
                 warmup_ratio=0.001,
                 step=[7000])

# Runner type
runner = dict(type='IterBasedRunner', max_iters=8000)

checkpoint_config = dict(interval=1000)

checkpoint_config = dict(interval=1000)
evaluation = dict(interval=1000)
