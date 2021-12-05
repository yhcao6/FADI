_base_ = 'fadi_split2_shot1_discrimination.py'

split = 2
shot = 10

model = dict(roi_head=dict(
    bbox_head=dict(split=split, loss_margin=dict(shot=shot, max_loss=0.5))))

unfreeze_layers = ('fc_cls', 'fc_reg', 'neck')

data = dict(train=dict(
    split=split,
    shot=shot,
))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[26000])

# Runner type
runner = dict(max_iters=30000)

checkpoint_config = dict(interval=1000)
evaluation = dict(interval=1000)

load_from = f'work_dirs/fadi_split{split}_shot{shot}_association/latest.pth'
