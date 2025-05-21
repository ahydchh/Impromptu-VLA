import os
import json
import torch
import math
from tqdm import tqdm
from nuscenes import NuScenes
from p_tqdm import p_map
from torch.utils.data import Dataset
from dataset_generation.ad_wrappers.nuscenes_3d import build_nuscenes_3d
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v0 import PromptNuScenesCoTV0
from dataset_generation.prompt_stages import *
from dataset_generation.prompt_stages.prompt_ego_status import PromptNuscenesEgoStatus


class PromptNuScenesCoTV1(PromptNuScenesCoTV0):
    # Directly mapping the input images to the output images
    def __init__(self, mode="train", length=None):
        super().__init__(mode=mode, length=length)
        
        # pipeline
        self.prompt_ego_status = PromptNuscenesEgoStatus(nuscenes=self.nuscenes_3d)
        self.pipelines = [
            self.prompt_metadata,
            self.prompt_ego_status,
            self.prompt_planning
        ]
        self.container_out_key_comb = [
            "planning"
        ]
