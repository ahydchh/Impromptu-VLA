from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v0 import PromptNuScenesCoTV0
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v1 import PromptNuScenesCoTV1
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v2 import PromptNuScenesCoTV2
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v3 import PromptNuScenesCoTV3
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v4 import PromptNuScenesCoTV4
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v5 import PromptNuScenesCoTV5
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v6 import PromptNuScenesCoTV6


DATASETS = {
    "nuscenes_train_v0": PromptNuScenesCoTV0,
    "nuscenes_train_v1": PromptNuScenesCoTV1,
    "nuscenes_train_v2": PromptNuScenesCoTV2,
    "nuscenes_train_v3": PromptNuScenesCoTV3,
    "nuscenes_train_v4": PromptNuScenesCoTV4,
    "nuscenes_train_v5": PromptNuScenesCoTV5,
    "nuscenes_train_v6": PromptNuScenesCoTV6,
}

if __name__ == '__main__':
    import os
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    assert args.dataset in DATASETS, f"Dataset {args.dataset} not found"
    dataset = DATASETS[args.dataset](mode="test")
    assert os.path.exists(args.jsonl_file), f"File {args.jsonl_file} not found"
   
    results = dataset.evaluate(jsonl_file=args.jsonl_file)
    # dump results to output_file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
