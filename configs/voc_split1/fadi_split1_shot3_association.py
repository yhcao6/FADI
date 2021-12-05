_base_ = 'fadi_split1_shot1_association.py'

split = 1
shot = 3
data = dict(train=dict(split=split, shot=shot),
            val=dict(split=split),
            test=dict(split=split))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1,
                 warmup_ratio=0.001,
                 step=[10000])

# Runner type
runner = dict(type='IterBasedRunner', max_iters=12000)

checkpoint_config = dict(interval=2000)
evaluation = dict(interval=2000)
