import sys
from operator import itemgetter

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

from fadi.datasets.few_shot_voc import BASE_CLASSES, NOVEL_CLASSES

split = int(sys.argv[1])
assert split in (1, 2, 3), 'Please input valid split, must be 1, 2 or 3'

base_classes = BASE_CLASSES[split]
novel_classes = NOVEL_CLASSES[split]
classes = base_classes + novel_classes

# fix some name alias
cls_map = dict()
for cls in classes:
    if cls == 'diningtable':
        cls_ = 'dining_table'
    elif cls == 'pottedplant':
        cls_ = 'flowerpot'
    elif cls == 'tvmonitor':
        cls_ = 'tv'
    elif cls == 'bicycle':
        cls_ = 'bike'
    else:
        cls_ = cls
    cls_map[cls] = cls_

semcor_ic = wordnet_ic.ic('ic-semcor.dat')
sim_dicts = dict()
for n_cls in novel_classes:
    n_cls_ = cls_map[n_cls]
    n_w = wn.synsets(n_cls_)[0]
    sims = []
    for b_cls in base_classes:
        b_cls_ = cls_map[b_cls]
        b_w = wn.synsets(b_cls_)[0]
        sim = n_w.lin_similarity(b_w,
                                 ic=semcor_ic)  # compute lin similarity here
        sim = round(sim, 3)
        sims.append((b_cls, sim))
    sims = sorted(sims,
                  key=itemgetter(1))[::-1][:5]  # display top5 similar classes
    sim_dicts[n_cls] = sims

for k, v in sim_dicts.items():
    print(k, v)
