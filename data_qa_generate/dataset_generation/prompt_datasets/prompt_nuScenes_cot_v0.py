import os
import json
import torch
import math
from tqdm import tqdm
from nuscenes import NuScenes
from p_tqdm import p_map
from torch.utils.data import Dataset
from dataset_generation.ad_wrappers.nuscenes_3d import build_nuscenes_3d
from dataset_generation.prompt_stages import *

class PromptNuScenesCoTV0(Dataset):
    # Directly mapping the input images to the output images
    def __init__(self, mode="train", length=None):
        self.nuscenes = build_nuscenes_3d(mode=mode)
        self.nuscenes_3d = NuScenes(version=self.nuscenes.version, dataroot=self.nuscenes.data_root, verbose=True)
        
        self.length = len(self.nuscenes)
        if length is not None:
            self.length = min(self.length, length)
        
        # defines in this version
        self.prompt_metadata = PromptNuscenesMetadata()
        self.prompt_planning = PromptNuscenesPlanning(nuscenes=self.nuscenes_3d)
        
        self.pipelines = [
            self.prompt_metadata,
            self.prompt_planning
        ]
        self.container_out_key_comb = [
            "planning"
        ]
        # defines in this version


    def cache_data(self, cache_filename):
        assert cache_filename.endswith(".json")
        # create cache file if not exists
        os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

        all_iters = list(range(len(self)))
        # Parallel processing even slower than serial processing. do not know why. by c7w
        # num_cpus = min(math.ceil(os.cpu_count() * 0.8), 16)
        # print("Using {} CPUs for caching".format(num_cpus))
        # all_jsons = p_map(self.__getitem__, all_iters, num_cpus=num_cpus, desc="Caching data")
        all_jsons = []
        for i in tqdm(all_iters, desc="Caching data"):
            all_jsons.append(self.__getitem__(i))
            
        # DUMP a json file!
        with open(cache_filename, "w") as f:
            f.write("[\n")
            all_len = len(all_jsons)
            for idx, json_obj in enumerate(all_jsons):
                f.write(json.dumps(json_obj))
                if idx != all_len - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("]")
        
        # cleanup all pipelines
        for pipeline in self.pipelines:
            if hasattr(pipeline, "cleanup"):
                pipeline.cleanup()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        batch = self.nuscenes[idx]
        
        ## pipeline
        container_out = {}
        for pipeline in self.pipelines:
            container_out = pipeline(container_out, batch)
        ## pipeline
        for key in self.container_out_key_comb:
            assert key in container_out["buffer_container"], "Key {} not found in container_out.buffer_container".format(key)
            container_out["messages"][1]["content"] += container_out["buffer_container"][key]

        # pop buffer container
        container_out.pop("buffer_container")
        return container_out

    def evaluate(self, jsonl_file):
        # load jsonl file to a list of dict
        predicted_data = []
        with open(jsonl_file, "r") as f:
            for line in f:
                predicted_data.append(json.loads(line))
        
        assert len(predicted_data) == len(self), "Length of predictions and dataset do not match: {} vs {}".format(len(predicted_data), len(self))
        # evaluate every single prediction
        for idx, pred in enumerate(predicted_data):
            # evaluate the prediction
            self.prompt_planning.evaluation_update(pred, self.nuscenes[idx])
        
        
        results = {}
        results["planning"] = self.prompt_planning.evaluation_compute()
        
        return results