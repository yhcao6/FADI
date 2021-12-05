import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead


@HEADS.register_module()
class CosineSimBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 fc_out_channels=1024,
                 scale=20.,
                 with_margin=False,
                 *args,
                 **kwargs):
        super(CosineSimBBoxHead,
              self).__init__(num_shared_convs=0,
                             num_shared_fcs=2,
                             num_cls_convs=0,
                             num_cls_fcs=0,
                             num_reg_convs=0,
                             num_reg_fcs=0,
                             fc_out_channels=fc_out_channels,
                             *args,
                             **kwargs)
        self.fc_cls = nn.Linear(self.cls_last_dim,
                                self.num_classes + 1,
                                bias=False)
        self.scale = scale
        self.with_margin = with_margin

    def forward(self, x, return_fc_feat=False):
        x = x.flatten(1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                               dim=1).unsqueeze(1).expand_as(
                                   self.fc_cls.weight.data)
        self.fc_cls.weight.data = self.fc_cls.weight.data.div(temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist
        bbox_preds = self.fc_reg(x)
        if return_fc_feat:
            return scores, bbox_preds, x_normalized
        return scores, bbox_preds

    def forward_cls(self, x):
        x = x.flatten(1)

        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                               dim=1).unsqueeze(1).expand_as(
                                   self.fc_cls.weight.data)
        self.fc_cls.weight.data = self.fc_cls.weight.data.div(temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist
        return scores

    def forward_bbox(self, x):
        x = x.flatten(1)

        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        bbox_preds = self.fc_reg(x)
        return bbox_preds

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
