import os
import json
import torch
import math
from tqdm import tqdm
from p_tqdm import p_map
from nuscenes import NuScenes
from torch.utils.data import Dataset
from dataset_generation.ad_wrappers.nuscenes_3d import build_nuscenes_3d
from dataset_generation.prompt_stages import *
from dataset_generation.prompt_stages.prompt_ego_status import PromptNuscenesEgoStatus
from dataset_generation.prompt_stages.prompt_meta_planning import PromptNuscenesMetaPlanning
from dataset_generation.prompt_stages.prompt_predict_obj import PromptNuscenesPredictObj
from dataset_generation.prompt_stages.prompt_scene_desc import PromptNuscenesSceneDesc
from dataset_generation.prompt_stages.prompt_perception_obj import PromptNuscenesPerceptionObj
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v3 import PromptNuScenesCoTV3

class PromptNuScenesCoTV5(PromptNuScenesCoTV3):
    # Directly mapping the input images to the output images
    def __init__(self, mode="train", cache_filename="", length=None):
        super().__init__(mode=mode, length=length)
        self.prompt_meta_planning = PromptNuscenesMetaPlanning(self.nuscenes_3d, "data/nuscenes/nuscenes_train_v2.json" if mode == "train" else "data/nuscenes/nuscenes_test_v2.json")
        self.pipelines = [
            self.prompt_metadata,
            self.prompt_ego_status,
            self.prompt_scene_desc,
            self.prompt_meta_planning,
            self.prompt_planning,
        ]
        self.container_out_key_comb = [
            "scene_desc", "meta_planning", "planning"
        ]


if __name__ == "__main__":
    prompt = PromptNuScenesCoTV5(mode="train", length=200)
    all_queries = []
    for idx in tqdm(range(len(prompt)), desc="Caching queries"):
        batch = prompt.nuscenes[idx]
        container_out = {}
        container_out = prompt.prompt_metadata(container_out, batch)
        container_out = prompt.prompt_ego_status(container_out, batch)
        query = prompt.prompt_meta_planning._cache_construct_query(container_out, batch)
        all_queries.append(query)
    
    import IPython; IPython.embed()
    from dataset_generation.prompt_stages.utils.external_query import construct_external_query
    external_helper = construct_external_query("Qwen/Qwen2.5-VL-72B-Instruct")
    
    from copy import deepcopy
    query = all_queries[74]
    images = deepcopy(query['images'])
    content = query['messages'][0]['content']
    helper_ret = external_helper.query_with_context(content, img=images)
    print(helper_ret)
    
    
    # helper_ret = external_helper.query_with_context(SCENE_DESC_PROMPT, img=container_out['images'][0])
    
    