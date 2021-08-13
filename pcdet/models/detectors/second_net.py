from .detector3d_template import Detector3DTemplate
import torch.nn


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.iter = 0
        self.count = 0
        self.huber = torch.nn.SmoothL1Loss(reduction='sum')
        self.loss_ago_sum = 0

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        x = batch_dict['spatial_features_2d']
        vx = batch_dict['pspatial_features_2d']
        mask = batch_dict['mask'].unsqueeze(1)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()
            num_nonezero = torch.sum(mask != 0)
            loss_ago = self.huber(x * mask, vx * mask) / num_nonezero
            self.iter += 1
            ret_dict = {
                'loss': loss + 0 * loss_ago
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
