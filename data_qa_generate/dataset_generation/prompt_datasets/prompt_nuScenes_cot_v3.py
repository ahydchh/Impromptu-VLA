import os
import json
import torch
import math
from tqdm import tqdm
from p_tqdm import p_map
from torch.utils.data import Dataset
from dataset_generation.ad_wrappers.nuscenes_3d import build_nuscenes_3d
from dataset_generation.prompt_stages import *
from dataset_generation.prompt_stages.prompt_road_agent_analysis import PromptNuscenesRoadAgentAnalysis
from dataset_generation.prompt_stages.prompt_ego_status import PromptNuscenesEgoStatus
from dataset_generation.prompt_stages.prompt_scene_desc import PromptNuscenesSceneDesc
from dataset_generation.prompt_stages.prompt_perception_obj import PromptNuscenesPerceptionObj
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v2 import PromptNuScenesCoTV2

class PromptNuScenesCoTV3(PromptNuScenesCoTV2):
    # Directly mapping the input images to the output images
    def __init__(self, mode="train", length=None):
        super().__init__(mode=mode, length=length)

        self.prompt_agent_analysis = PromptNuscenesRoadAgentAnalysis()
        self.pipelines = [
            self.prompt_metadata,
            self.prompt_ego_status,
            self.prompt_scene_desc,
            self.prompt_agent_analysis,
            self.prompt_planning,
        ]
        self.container_out_key_comb = [
            "scene_desc", "road_agent_analysis", "planning"
        ]
