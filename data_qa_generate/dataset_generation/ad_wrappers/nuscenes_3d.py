import os
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset
from dataset_generation.mmdet3d_plugin.datasets.builder import build_dataloader

def build_nuscenes_3d(mode="train"):
    cfg = Config.fromfile(
        os.path.join(os.path.dirname(__file__), "nuscenes_3d_config.py")
    )
    # train_dataset = build_dataset(cfg.data.train)
    # val_dataset = build_dataset(cfg.data.test)
    if mode == "train":
        dataset = build_dataset(cfg.data.train)
    else:
        dataset = build_dataset(cfg.data.test)
        
    return dataset

