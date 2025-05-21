import os
import json
import torch
import math
from tqdm import tqdm
from p_tqdm import p_map
from torch.utils.data import Dataset
from dataset_generation.ad_wrappers.nuscenes_3d import build_nuscenes_3d
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v1 import PromptNuScenesCoTV1
from dataset_generation.prompt_stages import *
from dataset_generation.prompt_stages.prompt_ego_status import PromptNuscenesEgoStatus
from dataset_generation.prompt_stages.prompt_scene_desc import PromptNuscenesSceneDesc

class PromptNuScenesCoTV2(PromptNuScenesCoTV1):
    # Directly mapping the input images to the output images
    def __init__(self, mode="train", cache_filename="", length=None):
        super().__init__(mode=mode, length=length)
        self.prompt_scene_desc = PromptNuscenesSceneDesc()
        self.pipelines = [
            self.prompt_metadata,
            self.prompt_ego_status,
            self.prompt_scene_desc,
            self.prompt_planning,
        ]
        self.container_out_key_comb = [
            "scene_desc", "planning"
        ]


    def cache_queries(self, query_filename, pipeline, max_len=8000):
        assert query_filename.endswith(".json")
        # cache all queries
        assert pipeline in self.pipelines, "Pipeline not in the list of pipelines"
        all_queries = []
        for idx in tqdm(range(len(self)), desc="Caching queries"):
            batch = self.nuscenes[idx]
            container_out = {}
            container_out = self.prompt_metadata(container_out, batch)
            query = pipeline._cache_construct_query(container_out, batch)
            all_queries.append(query)
        
        # divide the queries into chunks, and save them
        if len(all_queries) > max_len:
            num_chunks = math.ceil(len(all_queries) / max_len)
            chunk_size = math.ceil(len(all_queries) / num_chunks)
            for i in range(num_chunks):
                chunk_queries = all_queries[i * chunk_size: (i + 1) * chunk_size]
                chunk_filename = query_filename.replace(".json", "_{}.json".format(i))
                with open(chunk_filename, "w") as f:
                    json.dump(chunk_queries, f)
        else:
            with open(query_filename, "w") as f:
                json.dump(all_queries, f)

    def cache_responses(self, response_filename, pipeline, max_len=8000):
        assert pipeline in self.pipelines, "Pipeline not in the list of pipelines"
        assert response_filename.endswith(".jsonl")
        all_responses = []
        
        if len(self) > max_len:
            num_chunks = math.ceil(len(self) / max_len)
            for i in range(num_chunks):
                chunk_filename = response_filename.replace(".jsonl", "_{}.jsonl".format(i))
                with open(chunk_filename, "r") as f:
                    for line in f:
                        all_responses.append(json.loads(line))
        else:
            with open(response_filename, "r") as f:
                for line in f:
                    all_responses.append(json.loads(line))
        assert len(all_responses) == len(self), "Length of responses and dataset do not match: {} vs {}".format(len(all_responses), len(self))
        for idx, response in enumerate(tqdm(all_responses)):
            batch = self.nuscenes[idx]
            container_out = {}
            container_out = self.prompt_metadata(container_out, batch)
            pipeline._cache_construct_response(container_out, batch, response)
        pipeline.cleanup()
        return len(all_responses)
