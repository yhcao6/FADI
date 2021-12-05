_base_ = '../voc_split1/fadi_split1_shot1_association.py'

split = 3
shot = 1
data = dict(train=dict(split=split, shot=shot),
            val=dict(split=split),
            test=dict(split=split))

load_from = 'models/voc_split3_base.pth'

checkpoint_config = dict(interval=1000)
evaluation = dict(interval=1000)
