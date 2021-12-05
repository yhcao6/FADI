_base_ = 'fadi_split2_shot1_association.py'

split = 2
shot = 10
data = dict(train=dict(split=split, shot=shot),
            val=dict(split=split),
            test=dict(split=split))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1,
                 warmup_ratio=0.001,
                 step=[16000])

# Runner type
runner = dict(type='IterBasedRunner', max_iters=20000)

checkpoint_config = dict(interval=2000)
evaluation = dict(interval=2000)
