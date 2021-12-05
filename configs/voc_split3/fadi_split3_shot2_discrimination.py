_base_ = 'fadi_split3_shot1_discrimination.py'

split = 3
shot = 2

model = dict(roi_head=dict(
    bbox_head=dict(split=split, loss_margin=dict(shot=shot))))

data = dict(train=dict(
    split=split,
    shot=shot,
))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1,
                 warmup_ratio=0.001,
                 step=[7000])

# Runner type
runner = dict(max_iters=8000)

checkpoint_config = dict(interval=1000)
evaluation = dict(interval=1000)

load_from = f'work_dirs/fadi_split{split}_shot{shot}_association/latest.pth'
