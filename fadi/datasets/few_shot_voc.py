import copy
import os
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict

import numpy as np
from mmcv.runner.dist_utils import get_dist_info
from mmcv.utils import print_log
from mmdet.datasets import VOCDataset
from mmdet.datasets.builder import DATASETS

from .util import print_instances_class_histogram
from .voc_eval import eval_map

BASE_CLASSES = {
    1: [
        'aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep',
        'train', 'tvmonitor'
    ],
    2: [
        'bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'diningtable',
        'dog', 'motorbike', 'person', 'pottedplant', 'sheep', 'train',
        'tvmonitor'
    ],
    3: [
        'aeroplane', 'bicycle', 'bird', 'bottle', 'bus', 'car', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'train',
        'tvmonitor'
    ],
}

NOVEL_CLASSES = {
    1: ['bird', 'bus', 'cow', 'motorbike', 'sofa'],
    2: ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    3: ['boat', 'cat', 'motorbike', 'sheep', 'sofa'],
}

ASSIGN_DICT = {
    1: {
        'bird': 'dog',
        'bus': 'train',
        'cow': 'horse',
        'motorbike': 'bicycle',
        'sofa': 'chair'
    },
    2: {
        'aeroplane': 'boat',
        'bottle': 'car',
        'cow': 'sheep',
        'horse': 'dog',
        'sofa': 'chair'
    },
    3: {
        'boat': 'aeroplane',
        'cat': 'dog',
        'motorbike': 'bicycle',
        'sheep': 'cow',
        'sofa': 'chair'
    }
}


@DATASETS.register_module()
class FewShotVOCDataset(VOCDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    seeds = {
        1: dict(shot1=1, shot2=6, shot3=1, shot5=4, shot10=4),
        2: dict(shot1=5, shot2=5, shot3=1, shot5=3, shot10=0),
        3: dict(shot1=1, shot2=1, shot3=1, shot5=5, shot10=1),
    }

    def __init__(self, ann_file, shot, split, associate=False, **kwargs):
        self.shot = shot
        self.split = split
        self.associate = associate
        self.assign_dict = ASSIGN_DICT[split]
        self.base_classes = BASE_CLASSES[split]
        self.novel_classes = NOVEL_CLASSES[split]
        super(VOCDataset,
              self).__init__(ann_file=ann_file,
                             classes=self.base_classes + self.novel_classes,
                             **kwargs)

    def load_annotations(self, ann_file):
        rank, _ = get_dist_info()
        # copy from TFA
        classnames = self.CLASSES

        all_info_dict = {cls: defaultdict(list) for cls in classnames}
        dicts = []
        for shot in (1, 2, 3, 5, 10):
            if shot > self.shot:
                break
            seed = self.seeds[self.split][f'shot{shot}']
            fileids = {}
            split_dir = 'data/few_shot_voc_split'
            for cls in classnames:
                fn = f'box_{shot}shot_{cls}_train.txt'
                with open(os.path.join(split_dir, fn)) as f:
                    fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                    if isinstance(fileids_, str):
                        fileids_ = [fileids_]
                    fileids_ = [
                        fid.split('/')[-1].split('.jpg')[0] for fid in fileids_
                    ]
                    fileids[cls] = fileids_

            for cls, fileids_ in fileids.items():
                dicts_ = []
                for fileid in fileids_:
                    year = '2012' if '_' in fileid else '2007'
                    dirname = os.path.join('data/VOCdevkit/',
                                           'VOC{}'.format(year))
                    anno_file = os.path.join(dirname, 'Annotations',
                                             fileid + '.xml')
                    jpeg_file = os.path.join(dirname, 'JPEGImages',
                                             fileid + '.jpg')

                    tree = ET.parse(anno_file)

                    for i, obj in enumerate(tree.findall('object')):
                        r = {
                            'file_name': jpeg_file,
                            'image_id': fileid,
                            'height':
                            int(tree.findall('./size/height')[0].text),
                            'width': int(tree.findall('./size/width')[0].text),
                            'idx': i,
                        }
                        cls_ = obj.find('name').text
                        if cls != cls_:
                            continue
                        if i in all_info_dict[cls][r['file_name']]:
                            continue
                        bbox = obj.find('bndbox')
                        bbox = [
                            float(bbox.find(x).text)
                            for x in ['xmin', 'ymin', 'xmax', 'ymax']
                        ]
                        bbox[0] -= 1.0
                        bbox[1] -= 1.0

                        instances = [{
                            'category_id': classnames.index(cls),
                            'bbox': bbox,
                        }]
                        r['annotations'] = instances
                        dicts_.append(r)
                select_cls_dict = all_info_dict[cls]
                select_shots = []
                for _, v in select_cls_dict.items():
                    select_shots.extend(v)
                num_select_shots = len(select_shots)
                num_expected_shots = shot - num_select_shots

                # ensure all rank load same instances
                if len(dicts_) > int(num_expected_shots):
                    np.random.seed(seed)  # hack there
                    dicts_ = np.random.choice(dicts_,
                                              num_expected_shots,
                                              replace=False)
                    np.random.seed(rank)  # recover seed

                for info in dicts_:
                    all_info_dict[cls][info['file_name']].append(info['idx'])
                dicts.extend(dicts_)

        data_infos = []
        catid2cnts = defaultdict(lambda: 0)

        # each instance yield one image
        for info_dict in dicts:
            img_id = info_dict['image_id']
            filename = info_dict['file_name']
            height = info_dict['height']
            width = info_dict['width']
            img_info = dict(id=img_id,
                            filename=filename,
                            height=height,
                            width=width)

            assert len(info_dict['annotations']) == 1
            bboxes = info_dict['annotations'][0]['bbox']
            labels = info_dict['annotations'][0]['category_id']

            if self.associate:
                name = classnames[labels]
                # If a novel class, give it pseudo label
                if name in self.assign_dict:
                    pseudo_cat = self.assign_dict[name]
                    labels = self.CLASSES.index(pseudo_cat)

                if name in self.assign_dict.values():
                    continue

            bboxes = np.array([bboxes], dtype=np.float32).reshape(-1, 4)
            labels = np.array([labels], dtype=np.int64).reshape(-1)

            for label in labels:
                catid2cnts[label] += 1
            data_info = copy.deepcopy(img_info)
            data_info['annotations'] = dict(bboxes=bboxes, labels=labels)
            data_infos.append(data_info)

        cnts = [(self.CLASSES[label], cnt)
                for label, cnt in catid2cnts.items()]
        print_instances_class_histogram(cnts,
                                        self.CLASSES,
                                        total_img_nums=len(data_infos))
        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['annotations']

    def _filter_imgs(self, min_size=32):
        valid_inds = range(len(self.data_infos))
        return valid_inds


@DATASETS.register_module()
class FewShotVOCTestDataset(VOCDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, ann_file, split, associate=False, **kwargs):
        self.split = split
        self.base_classes = BASE_CLASSES[split]
        self.novel_classes = NOVEL_CLASSES[split]
        self.associate = associate
        self.assign_dict = ASSIGN_DICT[split]
        super(VOCDataset,
              self).__init__(ann_file=ann_file,
                             classes=self.base_classes + self.novel_classes,
                             **kwargs)
        if associate:
            self.cat2label = {}
            for cls in self.CLASSES:
                if cls in self.assign_dict:
                    pseudo_cat = self.assign_dict[cls]
                    self.cat2label[cls] = self.base_classes.index(pseudo_cat)
                else:
                    self.cat2label[cls] = self.base_classes.index(cls)

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            # only difference
            if self.associate and name in self.assign_dict:
                label = self.cat2label[self.assign_dict[name]]  # pseudo label
            else:
                label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if not self.test_mode:
                difficult = 0  # training keep difficult
            if self.associate and name in self.assign_dict.values():
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            elif difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            bboxes[:, :2] -= 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            bboxes_ignore[:, :2] -= 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(bboxes=bboxes.astype(np.float32),
                   labels=labels.astype(np.int64),
                   bboxes_ignore=bboxes_ignore.astype(np.float32),
                   labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        assert isinstance(iou_thrs, list)
        mean_aps = []
        for iou_thr in iou_thrs:
            print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            mean_ap, _ = eval_map(results,
                                  annotations,
                                  scale_ranges=None,
                                  iou_thr=iou_thr,
                                  dataset=self.CLASSES,
                                  logger=logger,
                                  mode='11points',
                                  base_classes=self.base_classes,
                                  novel_classes=self.novel_classes,
                                  cat2label=self.cat2label)
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        return eval_results
