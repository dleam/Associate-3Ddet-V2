# Associate-3Ddet-V2

Implementation of [Association-Guided 3D Point Cloud Object Detection Network](https://ieeexplore.ieee.org/document/9511841) (to be updated.)

## Requirements

#### Installation

Our code is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), please refer to
the [installation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) for OpenPCDet compilation.

## Data preparation

### NuScenes Dataset (to be updated)
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and
  organize the downloaded files as follows:
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval
├── pcdet
├── tools
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command:
```shell script
pip install nuscenes-devkit==1.0.5
```

* Generate the data infos by running the following command (it may take several hours):
```python
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
```


## Training & Testing

### Train a model

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```



### Test and evaluate the pretrained models
* Test with a pretrained model:
```shell script
python test.py --cfg_file ${CONFIG_FILE} --ckpt ${CKPT}
```


## Acknowledgements

We thank [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) for the great works and repos.
