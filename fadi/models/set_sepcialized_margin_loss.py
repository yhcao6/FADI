import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class SetSpecializedMarginLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 shot,
                 loss_base_margin_weight=None,
                 loss_novel_margin_weight=None,
                 loss_neg_margin_weight=0.001,
                 max_diff=1.,
                 max_loss=1.,
                 ignore_neg=False,
                 reduction='sum'):
        super(SetSpecializedMarginLoss, self).__init__()
        self.max_diff = max_diff
        self.max_loss = max_loss
        self.reduction = reduction

        # beta
        if loss_novel_margin_weight is None:
            self.loss_novel_margin_weight = 1. / shot
        else:
            self.loss_novel_margin_weight = loss_novel_margin_weight

        # alpha
        if loss_base_margin_weight is None:
            self.loss_base_margin_weight = self.loss_novel_margin_weight / 3.
        else:
            self.loss_base_margin_weight = loss_base_margin_weight

        # gamma
        self.loss_neg_margin_weight = loss_neg_margin_weight

        self.num_classes = num_classes
        self.ignore_neg = ignore_neg

    def forward(self, cls_score, labels, label_weights=None, **kwargs):
        losses = dict()

        num_base_classes = self.num_classes // 4 * 3
        base_inds = labels < num_base_classes
        novel_inds = (labels >= num_base_classes) & (labels < self.num_classes)
        base_labels = labels[base_inds]
        novel_labels = labels[novel_inds]
        scores = cls_score.softmax(-1)
        base_scores = scores[base_inds, base_labels]
        novel_scores = scores[novel_inds, novel_labels]

        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        # base margin
        num_base = base_scores.size(0)
        if num_base > 0:
            loss_base_margin = []
            for label in base_labels.unique():
                l_inds = labels == label
                l_scores = scores[l_inds, label]
                l_all_scores = scores[l_inds]
                mask = torch.ones_like(l_all_scores).bool()
                mask[:, label] = False
                if self.ignore_neg:
                    mask[:, -1] = False
                l_other_scores = l_all_scores[mask].reshape(
                    l_scores.size(0), -1)
                diff = l_scores[:, None] - l_other_scores
                diff.clamp_(min=1e-7, max=self.max_diff)
                loss_base_margin.append(-diff.log())
            loss_base_margin = torch.cat(loss_base_margin)
            if self.reduction == 'sum':
                loss_base_margin = loss_base_margin.sum().div(avg_factor)
            else:
                loss_base_margin = loss_base_margin.mean()
            loss_base_margin *= self.loss_base_margin_weight
        else:
            loss_base_margin = cls_score.sum() * 0.
        losses['loss_base_margin'] = loss_base_margin

        # novel margin
        num_novel = novel_scores.size(0)
        if num_novel > 0:
            loss_novel_margin = []
            for label in novel_labels.unique():
                l_inds = labels == label
                l_scores = scores[l_inds, label]
                l_all_scores = scores[l_inds]
                mask = torch.ones_like(l_all_scores).bool()
                mask[:, label] = False
                if self.ignore_neg:
                    mask[:, -1] = False
                l_other_scores = l_all_scores[mask].reshape(
                    l_scores.size(0), -1)
                diff = l_scores[:, None] - l_other_scores
                diff.clamp_(min=1e-7, max=self.max_diff)
                loss_novel_margin.append(-diff.log())
            loss_novel_margin = torch.cat(loss_novel_margin)
            if self.reduction == 'sum':
                loss_novel_margin = loss_novel_margin.sum().div(avg_factor)
            else:
                loss_novel_margin = loss_novel_margin.mean()
            loss_novel_margin *= self.loss_novel_margin_weight
        else:
            loss_novel_margin = cls_score.sum() * 0.
        losses['loss_novel_margin'] = loss_novel_margin

        # neg margin
        neg_inds = labels == self.num_classes
        neg_scores = scores[neg_inds, -1]
        neg_other_scores = scores[neg_inds, :-1]
        diff = neg_scores[:, None] - neg_other_scores
        diff.clamp_(min=1e-7, max=self.max_diff)
        loss_neg_margin = -diff.log()
        if self.reduction == 'sum':
            loss_neg_margin = loss_neg_margin.sum().div(avg_factor)
        else:
            loss_neg_margin = loss_neg_margin.mean()
        loss_neg_margin *= self.loss_neg_margin_weight
        losses['loss_neg_margin'] = loss_neg_margin

        if self.reduction == 'sum':
            if 'loss_novel_margin' in losses:
                losses['loss_novel_margin'].clamp_(max=self.max_loss)
            if 'loss_base_margin' in losses:
                losses['loss_base_margin'].clamp_(max=self.max_loss)
            if 'loss_neg_margin' in losses:
                losses['loss_neg_margin'].clamp_(max=self.max_loss)

        return losses
