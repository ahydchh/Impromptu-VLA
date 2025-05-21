from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v0 import PromptNuScenesCoTV0
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v1 import PromptNuScenesCoTV1
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v2 import PromptNuScenesCoTV2
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v3 import PromptNuScenesCoTV3
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v4 import PromptNuScenesCoTV4
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v5 import PromptNuScenesCoTV5
from dataset_generation.prompt_datasets.prompt_nuScenes_cot_v6 import PromptNuScenesCoTV6

if __name__ == '__main__':

    dataset = PromptNuScenesCoTV6(mode="train")
    dataset.cache_data("data/nuscenes/nuscenes_train_v6.json")
    dataset = PromptNuScenesCoTV6(mode="test")
    dataset.cache_data("data/nuscenes/nuscenes_test_v6.json")
    
    print("Done")