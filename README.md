# Few-Shot Object Detection via Association and DIscrimination

Code release of our NeurIPS 2021 paper: [Few-Shot Object Detection via Association and DIscrimination](https://arxiv.org/abs/2111.11656).

![FSCE Figure](https://i.imgur.com/YLpjAMa.png)

### Bibtex

```
@inproceedings{cao2021few,
  title={Few-Shot Object Detection via Association and DIscrimination},
  author={Cao, Yuhang and Wang, Jiaqi and Jin, Ying and Wu, Tong and Chen, Kai and Liu, Ziwei and Lin, Dahua},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

Arxiv: https://arxiv.org/abs/2111.11656

## Install dependencies

* Create a new environment: ```conda create -n fadi python=3.8 -y```
* Active the newly created environment: ```conda activate fadi```
* Install [PyTorch](https://pytorch.org/get-started/locally/) and [torchvision](https://github.com/pytorch/vision/): ```conda install pytorch=1.7 torchvision cudatoolkit=10.2 -c pytorch -y```
* Install [MMDetection](https://github.com/open-mmlab/mmdetection): ```pip install mmdet==2.11.0```
* Install [MMCV](https://github.com/open-mmlab/mmdetection): ```pip install mmcv==1.2.5```
* Install [MMCV-Full](https://github.com/open-mmlab/mmdetection): ```pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.0/index.html```

Note:
* Only tested on MMDet==2.11.0, MMCV==1.2.5, it may not be consistent with other versions.
* The above instructions use CUDA 10.2, make sure you install the correct PyTorch, Torchvision and MMCV-Full that are consistent with your CUDA version.



## Prepare dataset

We follow exact the same split with [TFA](https://github.com/ucbdrive/few-shot-object-detection),
please download the dataset and split files as follows:

* Download [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
* Download [split files](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EV4lUT3Wrk1NrQMziWgu6awBInAPyY0KvROf5IEv_ly8TQ?e=MdbSXG)

Create a directory `data` in the root directory, and the expected structure for `data` directory:
```
data/
    VOCdevkit
    few_shot_voc_split
```


## Training & Testing

### Base Training

FADI share the same base training stage with [TFA](https://github.com/ucbdrive/few-shot-object-detection),
we directly convert the corresponding checkpoints from TFA in Detectron2 format to MMDetection format,
please download the base training checkpoints following the table.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Split</th>
<th valign="bottom"><br/>AP50</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr><td align="left">Base Model</td>
<td align="center">1</td>
<td align="center">80.8</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EVgGzmEwlT9LvfC9GhwkWYoBR4OjfZ8-U4jNnbzKf8l7mw">model</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EUfhrL4XUotNj110vzt6LbYBRdpy_5j5jLPUa87wX7jZZQ?e=tyhYCk">surgery</a></td>
</tr>

<tr><td align="left">Base Model</td>
<td align="center">2</td>
<td align="center">81.9</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EUo_SbsGnLJDvEwX3iMuJKwB5_GcRBYJ3gZsgB3BRnnUCg?e=7F3RJQ">model</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EaRsKWxDc7FAtedQgTSo49sBrg0U3vr-eMpyWawc4PV3LQ?e=cYOi6k">surgery</a></td>
</tr>

<tr><td align="left">Base Model</td>
<td align="center">3</td>
<td align="center">82.0</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EacaCwrJJDtIrV47_uQe6YsB6JNpgwB5idQBGbGKT5Bdcg?e=l7tFVl">model</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/ETQgNDMAG99PmnmPqIL-7_0BreZR5ZAtT8vEs0cV_oAKTA?e=gUJk13">surgery</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

Create a directory `models` in the root directory, and the expected structure for `models` directory:
```
models/
    voc_split1_base.pth
    voc_split1_base_surgery.pth
    voc_split2_base.pth
    voc_split2_base_surgery.pth
    voc_split3_base.pth
    voc_split3_base_surgery.pth
```

### Few-Shot Fine-tuning

FADI divides the few-shot fine-tuning stage into two steps, *ie*, **association** and **discrimination**,

Suppose we want to train a model for Pascal VOC split1, shot1 with 8 GPUs

#### 1. Step 1: Association.

Getting the assigning scheme of the split:

```
python tools/associate.py 1
```

Aligning the feature distribution of the **associated** base and novel classes:

```
./tools/dist_train.sh configs/voc_split1/fadi_split1_shot1_association.py 8
```

#### 2. Step 2: Discrimination

Building a discriminate feature space for novel classes with **disentangling** and **set-specialized margin loss**:
```
./tools/dist_train.sh configs/voc_split1/fadi_split1_shot1_discrimination.py 8
```

#### Holistically Training:

We also provide you a script [tools/fadi_finetune.sh](tools/fadi_finetune.sh) to holistically train a model for a specific split/shot by running:

```
./tools/fadi_finetune.sh 1 1
```

#### Evaluation

To evaluate the trained models, run

```
./tools/dist_test.sh configs/voc_split1/fadi_split1_shot1_discrimination.py [checkpoint] 8 --eval mAP --out res.pkl
```

## Model Zoo

### Pascal VOC split 1

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Shot</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr>
<td align="center">1</td>
<td align="center">50.6</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EcgQTZHX8IZMvKeWwdDyL44Bd0YJp5fTsFpRWs73dUGRaQ?e=uNUaH9">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EVGekTe7YdhBgMvbBLHY0EgBUeREeWTLQaRJLyFOkSJm-w?e=ncjfoB">discrimination</a></td>
</tr>

<tr>
<td align="center">2</td>
<td align="center">54.8</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/ERaxA8nwqqdPoCjZtzdR4hQB0LtL7eqBqCPsY-Dj34NWxw?e=951c5r">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EcDhd8a_yL1Kr9M34QcNtfgBnFDZzhKWrs98ethZ5RU4Dg?e=VcSqVB">discrimination</a></td>
</tr>

<tr>
<td align="center">3</td>
<td align="center">54.1</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EdeTfvipMF5FmnOOr4vkUKoBe_sje83H2cnyk_dlPHUGWA?e=mDiyQb">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EU51m6loM1NFkArr_hmnuKQBGaIxn6Px0OpeHDOgXtLLcw?e=PWaT8P">discrimination</a></td>
</tr>

<tr>
<td align="center">5</td>
<td align="center">59.4</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EeWXndLHGbNBoocfsoti3m0BUF4L7VSyBLNP93QPmM_fWA">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EZKO6O2s7ydEjveI3sa70NQBQawa76uSGv_jA4asVVdU-w?e=iEbHz5">discrimination</a></td>
</tr>

<tr>
<td align="center">10</td>
<td align="center">63.5</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EUB3yRqrUMRLvWXDeO3a4zEBDFMPgu0gWVx6x-zT8QesUA?e=PEq8AA">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EaxxBvz7YYFEmLyzvooS4pMB7dfI5rXL8NXu3C-BLOd65Q?e=R0mMgE">discrimination</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

### Pascal VOC split 2

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Shot</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr>
<td align="center">1</td>
<td align="center">30.5</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EaWKcdAqvC1Ejm8fzByhapABMuNpB53tSENaX_9Ks0j1vg?e=pTxQjX">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/Ec2RomIlrKlMlYgP9ByZrrUBIlSRme5s4qRrY257KpY2Eg?e=S2pWeY">discrimination</a></td>
</tr>

<tr>
<td align="center">2</td>
<td align="center">35.1</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EVhBOfu0QspNqNyP61byKScBScfzi64Nm9dL_J60SGc2aA?e=aKV94s">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EfGx1hBbEUtEgpm4t2FR6JQB9z_xVI76zyhGP_iBIKtykg?e=5fkylX">discrimination</a></td>
</tr>

<tr>
<td align="center">3</td>
<td align="center">40.3</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EU9436vXkLxFh7lQ62pF_pMBVY1eDZ8_zsf5G0FyO4Kxlw?e=b9lAzv">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EZ0V4TPoyyRPuYR_BNiLYd0B7EUxjyd3GoKdAaolVCSbjQ?e=pnry0t">discrimination</a></td>
</tr>

<tr>
<td align="center">5</td>
<td align="center">42.9</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/ETE1cXT0L3tDqftPl5TRSQoBH4voFnWNnJXJk1tECvnrHg?e=gZ2EEe">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EQ4JWJEPxfFIsINdJEA-XiwBnujzoc5i5e0R_e0pG6VnAQ?e=ScEFej">discrimination</a></td>
</tr>

<tr>
<td align="center">10</td>
<td align="center">48.3</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/Ea0ebSt3LXJOmwPNUy7pnNcB9YyH2rzLKSM33Scp3LH_fw?e=RYro0U">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/ERW0_hWDAe1IlbN9RQomYyYBnIbkbpW8BRqyKwA50AuNww?e=CYQYpS">discrimination</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>

### Pascal VOC split 3

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Shot</th>
<th valign="bottom"><br/>nAP50</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->

<tr>
<td align="center">1</td>
<td align="center">45.7</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EV1gPw915YxMo_xyArZ7pBwBO9seyl8Krwv9UGIM6ihB7A?e=heRZF1">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/ESxxmVrh4l9Ghlk1HvuhGvYBetNxTgzKFX30z-IpURYLkA?e=zkewR1">discrimination</a></td>
</tr>

<tr>
<td align="center">2</td>
<td align="center">49.4</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EUV_rhgyvBBEoCxLTRFiy4EB0hRNzUbLMzSz4JHy5Vi_dA?e=5w9bZw">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EdH126l3klpFrJ3_2WgC14cB6BVCcvzjVmTGLmjhuGvvlA?e=pVl2p2">discrimination</a></td>
</tr>

<tr>
<td align="center">3</td>
<td align="center">49.4</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/ETCD0lJPucBMoCdzEmpVuFYBro4Flcsq6Ma-br-y69SF8g?e=fPd3do">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EdBEoyBLftRLpGeQtV5VqkEBqIRfjVdPAuU5gJoj6KCouw?e=iycmLq">discrimination</a></td>
</tr>

<tr>
<td align="center">5</td>
<td align="center">55.1</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EcSM5ZAvnwtPt0G2Ztm0nq0Bgv61ievZUorA1cju6DihEA?e=BNoWyL">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EbluJOyHyMBLthk-RTNVWSkBAyvAauZtxNQuiXNkjtd1Og?e=6nSqus">discrimination</a></td>
</tr>

<tr>
<td align="center">10</td>
<td align="center">59.3</td>
<td align="center"><a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EXqwm8zyYktNuROZxHssZ1cBSuuPrerBjEFd3mQVHWDsrw?e=IbAwHG">association</a>
      &nbsp;|&nbsp;<a href="https://mycuhk-my.sharepoint.com/:u:/g/personal/1155092180_link_cuhk_edu_hk/EZfy4Utv6ddBuQLanmBPJ5EB75jjQq6w8B5f8pvvuWB7bQ?e=x6Cf6x">discrimination</a></td>
</tr>

<!-- END OF TABLE BODY -->
</tbody></table>
