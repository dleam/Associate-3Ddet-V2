import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                             self.nx)
        batch_dict['spatial_features'] = batch_spatial_features

        # -----------------
        ppillar_features, pcoords = batch_dict['ppillar_features'], batch_dict['pvoxel_coords']
        pbatch_spatial_features = []
        pbatch_size = pcoords[:, 0].max().int().item() + 1
        for pbatch_idx in range(pbatch_size):
            pspatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=ppillar_features.dtype,
                device=ppillar_features.device)

            pbatch_mask = pcoords[:, 0] == pbatch_idx
            pthis_coords = pcoords[pbatch_mask, :]
            pindices = pthis_coords[:, 1] + pthis_coords[:, 2] * self.nx + pthis_coords[:, 3]
            pindices = pindices.type(torch.long)
            ppillars = ppillar_features[pbatch_mask, :]
            ppillars = ppillars.t()
            pspatial_feature[:, pindices] = ppillars
            pbatch_spatial_features.append(pspatial_feature)

        pbatch_spatial_features = torch.stack(pbatch_spatial_features, 0)
        pbatch_spatial_features = pbatch_spatial_features.view(pbatch_size, self.num_bev_features * self.nz, self.ny,
                                                               self.nx)
        batch_dict['pspatial_features'] = pbatch_spatial_features
        # -----------------

        return batch_dict